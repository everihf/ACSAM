from __future__ import annotations

import argparse
import csv
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
EXAMPLE_DIR = ROOT_DIR / "example"

for path in (CURRENT_DIR, ROOT_DIR, EXAMPLE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sam import SAM

from adaptive_curriculum import AdaptiveCurriculumConfig, AdaptiveCurriculumLearning

Cifar = None
Cifar100CNN = None
WideResNet = None
cifar_resnet18 = None
cifar_resnet32 = None
smooth_crossentropy = None
disable_running_stats = None
enable_running_stats = None
initialize = None
Log = None
build_logger = None
StepLR = None
evaluate_accuracy = None
pretrain_teacher_model = None


def _load_training_dependencies() -> None:
    global Cifar
    global Cifar100CNN
    global WideResNet
    global cifar_resnet18
    global cifar_resnet32
    global smooth_crossentropy
    global disable_running_stats
    global enable_running_stats
    global initialize
    global Log
    global build_logger
    global StepLR
    global evaluate_accuracy
    global pretrain_teacher_model

    if Cifar is not None:
        return

    from data.cifar import Cifar as _Cifar
    from model.cifar100_cnn import Cifar100CNN as _Cifar100CNN
    from model.cifar_resnet import cifar_resnet18 as _cifar_resnet18, cifar_resnet32 as _cifar_resnet32
    from model.smooth_cross_entropy import smooth_crossentropy as _smooth_crossentropy
    from model.wide_res_net import WideResNet as _WideResNet
    from utility.bypass_bn import (
        disable_running_stats as _disable_running_stats,
        enable_running_stats as _enable_running_stats,
    )
    from utility.initialize import initialize as _initialize
    from utility.log import Log as _Log, build_logger as _build_logger
    from utility.step_lr import StepLR as _StepLR
    from utility.teacher_model import (
        evaluate_accuracy as _evaluate_accuracy,
        pretrain_teacher_model as _pretrain_teacher_model,
    )

    Cifar = _Cifar
    Cifar100CNN = _Cifar100CNN
    WideResNet = _WideResNet
    cifar_resnet18 = _cifar_resnet18
    cifar_resnet32 = _cifar_resnet32
    smooth_crossentropy = _smooth_crossentropy
    disable_running_stats = _disable_running_stats
    enable_running_stats = _enable_running_stats
    initialize = _initialize
    Log = _Log
    build_logger = _build_logger
    StepLR = _StepLR
    evaluate_accuracy = _evaluate_accuracy
    pretrain_teacher_model = _pretrain_teacher_model


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class CurriculumBatchStream:
    """Yield one adaptive-curriculum batch at a time for a fixed epoch length."""

    def __init__(self, curriculum: AdaptiveCurriculumLearning, batch_size: int, num_batches: int):
        self.curriculum = curriculum
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.curriculum.sample_batch(self.batch_size)


def build_model(args, model_name: str, num_classes: int) -> torch.nn.Module:
    _load_training_dependencies()
    if model_name == "wrn":
        return WideResNet(
            args.depth,
            args.width_factor,
            args.dropout,
            in_channels=3,
            labels=num_classes,
        )
    if model_name == "cifar100_cnn":
        return Cifar100CNN(
            num_classes=num_classes,
            activation=args.cifar100_activation,
            dropout_1_rate=args.cifar100_dropout1,
            dropout_2_rate=args.cifar100_dropout2,
            batch_norm=args.cifar100_batch_norm,
        )
    if model_name == "resnet18":
        return cifar_resnet18(num_classes=num_classes)
    if model_name == "resnet32":
        return cifar_resnet32(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def _extract_model_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for nested_key in ("state_dict", "model_state_dict", "teacher_state_dict"):
            nested_state = checkpoint_obj.get(nested_key)
            if isinstance(nested_state, dict):
                checkpoint_obj = nested_state
                break
    if not isinstance(checkpoint_obj, dict):
        raise TypeError(f"Expected checkpoint to contain a state_dict-like dict, got {type(checkpoint_obj)!r}.")

    normalized_state = {}
    for key, value in checkpoint_obj.items():
        normalized_key = key[7:] if isinstance(key, str) and key.startswith("module.") else key
        normalized_state[normalized_key] = value
    return normalized_state


def _get_classifier_param_keys(model):
    classifier_weight_key = None
    classifier_bias_key = None
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            classifier_weight_key = f"{module_name}.weight" if module_name else "weight"
            classifier_bias_key = f"{module_name}.bias" if module.bias is not None else None
    return classifier_weight_key, classifier_bias_key


def _maybe_adapt_teacher_checkpoint_to_subset(checkpoint_state, teacher_model, dataset, checkpoint_path, logger):
    classifier_weight_key, classifier_bias_key = _get_classifier_param_keys(teacher_model)
    if classifier_weight_key is None:
        return checkpoint_state

    source_weight = checkpoint_state.get(classifier_weight_key)
    target_state = teacher_model.state_dict()
    target_weight = target_state.get(classifier_weight_key)
    if source_weight is None or target_weight is None:
        return checkpoint_state
    if tuple(source_weight.shape) == tuple(target_weight.shape):
        return checkpoint_state

    train_dataset = getattr(dataset, "train_dataset", None)
    subset_fine_indices = getattr(train_dataset, "selected_fine_indices", None)
    if subset_fine_indices is None:
        return checkpoint_state

    subset_fine_indices = [int(idx) for idx in subset_fine_indices]
    target_num_classes = int(target_weight.shape[0])
    source_num_classes = int(source_weight.shape[0])
    if len(subset_fine_indices) != target_num_classes:
        return checkpoint_state
    if not subset_fine_indices or max(subset_fine_indices) >= source_num_classes:
        return checkpoint_state

    adapted_state = dict(checkpoint_state)
    row_indices = torch.as_tensor(subset_fine_indices, dtype=torch.long, device=source_weight.device)
    adapted_state[classifier_weight_key] = source_weight.index_select(0, row_indices)

    source_bias = checkpoint_state.get(classifier_bias_key) if classifier_bias_key is not None else None
    if classifier_bias_key is not None and source_bias is not None:
        bias_indices = torch.as_tensor(subset_fine_indices, dtype=torch.long, device=source_bias.device)
        adapted_state[classifier_bias_key] = source_bias.index_select(0, bias_indices)

    subset_class_names = list(getattr(train_dataset, "classes", []))
    logger.info(
        "Adapted teacher checkpoint %s classifier from %d classes to subset classes %s (fine indices=%s).",
        checkpoint_path,
        source_num_classes,
        subset_class_names if subset_class_names else f"{target_num_classes} classes",
        subset_fine_indices,
    )
    return adapted_state


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def build_teacher_model(args, student_model, num_classes: int, device: torch.device, logger) -> torch.nn.Module:
    teacher_model_name = args.teacher_model or args.model
    use_custom_teacher_arch = any(
        value is not None
        for value in (args.teacher_depth, args.teacher_dropout, args.teacher_width_factor)
    )
    if teacher_model_name == "wrn" and use_custom_teacher_arch:
        teacher_depth = args.teacher_depth if args.teacher_depth is not None else args.depth
        teacher_dropout = args.teacher_dropout if args.teacher_dropout is not None else args.dropout
        teacher_width_factor = (
            args.teacher_width_factor if args.teacher_width_factor is not None else args.width_factor
        )
        teacher_model = WideResNet(
            teacher_depth,
            teacher_width_factor,
            teacher_dropout,
            in_channels=3,
            labels=num_classes,
        )
    else:
        if args.teacher_model is None:
            teacher_model = deepcopy(student_model)
            logger.info("Teacher architecture defaults to a deepcopy of student model.")
        else:
            teacher_model = build_model(args, teacher_model_name, num_classes)
    teacher_model = teacher_model.to(device)
    logger.info("Teacher architecture: %s", teacher_model_name)
    return teacher_model


def prepare_teacher_model(args, student_model, dataset, device, logger, checkpoint_dir: Path, run_name: str):
    teacher_model = build_teacher_model(args, student_model, len(dataset.classes), device, logger)
    if args.teacher_checkpoint:
        checkpoint_path = resolve_path(args.teacher_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {checkpoint_path}. "
                "Please verify --teacher_checkpoint."
            )
        teacher_state = torch.load(checkpoint_path, map_location=device)
        teacher_state = _extract_model_state_dict(teacher_state)
        teacher_state = _maybe_adapt_teacher_checkpoint_to_subset(
            checkpoint_state=teacher_state,
            teacher_model=teacher_model,
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )
        teacher_model.load_state_dict(teacher_state)
        logger.info("Loaded teacher checkpoint from %s", checkpoint_path)
    else:
        teacher_log_path = CURRENT_DIR / "logs" / f"{run_name}_teacher.log"
        teacher_log_path.parent.mkdir(parents=True, exist_ok=True)
        teacher_logger = build_logger("acl.train.teacher", teacher_log_path)
        teacher_best_checkpoint_path = checkpoint_dir / f"{run_name}_teacher_model.pt"
        teacher_model = pretrain_teacher_model(
            teacher_model=teacher_model,
            train_loader=dataset.train,
            test_loader=dataset.test,
            args=args,
            device=device,
            logger=teacher_logger,
            best_checkpoint_path=teacher_best_checkpoint_path,
        )
        if args.save_teacher_checkpoint:
            logger.info("Saved pretrained teacher checkpoint to %s", teacher_best_checkpoint_path)

    teacher_val_accuracy = evaluate_accuracy(teacher_model, dataset.test, device)
    logger.info(
        "Teacher validation accuracy before student training: %.2f%%",
        teacher_val_accuracy * 100,
    )
    return teacher_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model with the standalone ACL implementation.")
    parser.add_argument("--adaptive", default=True, type=parse_bool, help="Enable ASAM behavior when --optimizer sam.")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=[
            "cifar10",
            "cifar100",
            "cifar100_aquatic_mammals",
            "cifar100_small_mammals",
            "cifar100_household_electrical_devices",
        ],
    )
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--use_data_augmentation", default=True, type=parse_bool)
    parser.add_argument("--model", default="resnet18", type=str, choices=["wrn", "cifar100_cnn", "resnet18", "resnet32"])
    parser.add_argument("--depth", default=16, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--width_factor", default=8, type=int)
    parser.add_argument("--cifar100_activation", default="elu", type=str, choices=["elu", "relu", "gelu"])
    parser.add_argument("--cifar100_dropout1", default=0.25, type=float)
    parser.add_argument("--cifar100_dropout2", default=0.5, type=float)
    parser.add_argument("--cifar100_batch_norm", default=False, type=parse_bool)
    parser.add_argument("--teacher_model", default=None, type=str, choices=["wrn", "cifar100_cnn", "resnet18", "resnet32"])
    parser.add_argument("--teacher_depth", default=None, type=int)
    parser.add_argument("--teacher_dropout", default=None, type=float)
    parser.add_argument("--teacher_width_factor", default=None, type=int)
    parser.add_argument("--teacher_optimizer", default="sgd", type=str, choices=["sam", "sgd"])
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sam", "sgd"])
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--rho", default=2.0, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--curriculum_strategy", default="adaptive", type=str, choices=["none", "adaptive"])
    parser.add_argument("--teacher_checkpoint", default="", type=str)
    parser.add_argument("--pace_p", default=0.04, type=float)
    parser.add_argument("--pace_q", default=1.9, type=float)
    parser.add_argument("--pace_r", default=100, type=int)
    parser.add_argument("--inv", default=50, type=int)
    parser.add_argument("--alpha", default=-0.01, type=float)
    parser.add_argument("--difficulty_warmup_batches", default=150, type=int)
    parser.add_argument("--adaptive_teacher_source", default="teacher_model", type=str, choices=["teacher_model", "inception_svm"])
    parser.add_argument("--adaptive_loader_mode", default="stream", type=str, choices=["stream", "epoch_subset"])
    parser.add_argument("--lambda1", default=0.01, type=float)
    parser.add_argument("--lambda1_decay", default=None, type=float)
    parser.add_argument("--bottom_lambda1", default=0.0, type=float)
    parser.add_argument("--fixed_balance_order", default=True, type=parse_bool)
    parser.add_argument("--metrics_dir", default="metrics", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--run_name", default="", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_teacher_checkpoint", default=True, type=parse_bool)
    parser.add_argument("--save_student_checkpoint", default=True, type=parse_bool)
    parser.add_argument("--device", default="", type=str, help="Optional torch device override, e.g. cuda:0 or cpu.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.curriculum_strategy == "adaptive" and args.adaptive_teacher_source != "teacher_model":
        raise NotImplementedError(
            "ACL/train.py currently supports only --adaptive_teacher_source teacher_model."
        )
    _load_training_dependencies()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    pin_memory = device.type == "cuda"

    initialize(args, seed=args.seed)

    start_dt = datetime.now(ZoneInfo("Asia/Shanghai"))
    log_prefix = start_dt.strftime("%m-%d_%H-%M-%S")
    run_name = args.run_name or (
        f"{log_prefix}_acl-{args.curriculum_strategy}_{args.dataset}-{args.model}_seed{args.seed}"
    )

    log_dir = CURRENT_DIR / args.log_dir
    metrics_dir = CURRENT_DIR / args.metrics_dir
    checkpoint_dir = CURRENT_DIR / args.checkpoint_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger("acl.train", log_dir / f"{run_name}.log")
    logger.info("Run name: %s", run_name)
    logger.info("Device: %s", device)
    logger.info("Curriculum strategy: %s", args.curriculum_strategy)

    dataset = Cifar(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=args.dataset,
        use_data_augmentation=args.use_data_augmentation,
    )
    logger.info(
        "Dataset=%s, classes=%d, train_size=%d, test_size=%d",
        args.dataset,
        len(dataset.classes),
        len(dataset.train_dataset),
        len(dataset.test_dataset),
    )

    model = build_model(args, args.model, len(dataset.classes)).to(device)
    curriculum = None
    if args.curriculum_strategy == "adaptive":
        teacher_model = prepare_teacher_model(
            args=args,
            student_model=model,
            dataset=dataset,
            device=device,
            logger=logger,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
        )

        curriculum_config = AdaptiveCurriculumConfig(
            pace_p=args.pace_p,
            pace_q=args.pace_q,
            pace_r=args.pace_r,
            inv=args.inv,
            alpha=args.alpha,
            lambda_kl=args.lambda1,
            lambda_kl_decay=args.lambda1_decay,
            lambda_kl_min=args.bottom_lambda1,
            difficulty_warmup_batches=args.difficulty_warmup_batches,
            keep_class_balance=args.fixed_balance_order,
            score_mode="cross_entropy",
        )
        curriculum = AdaptiveCurriculumLearning(
            train_dataset=dataset.train_dataset,
            num_classes=len(dataset.classes),
            device=device,
            config=curriculum_config,
            teacher_model=teacher_model,
        )
        curriculum.initialize(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    else:
        if args.teacher_checkpoint or args.teacher_model is not None:
            logger.info(
                "Ignoring teacher-related options because curriculum_strategy=none."
            )

    if args.optimizer == "sam":
        optimizer = SAM(
            model.parameters(),
            torch.optim.SGD,
            rho=args.rho,
            adaptive=args.adaptive,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    log = Log(log_each=50, logger=logger)

    steps_per_epoch = len(dataset.train)
    metrics_path = metrics_dir / f"{run_name}.csv"
    best_checkpoint_path = checkpoint_dir / f"{run_name}_student_best.pt"
    val_curve = []
    best_val_accuracy = float("-inf")
    best_epoch = -1
    cumulative_batches = 0

    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "cumulative_batches", "elapsed_seconds", "val_accuracy", "pool_size", "curriculum_finished"],
        )
        writer.writeheader()

    train_start_perf = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start_perf = time.perf_counter()
        model.train()

        if curriculum is None:
            distillation_enabled = False
            train_loader = dataset.train
            pool_size = ""
            curriculum_finished = ""
        elif curriculum.curriculum_finished:
            distillation_enabled = False
            train_loader = curriculum.build_full_dataloader(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            pool_size = curriculum.current_pool_size()
            curriculum_finished = curriculum.curriculum_finished
        elif args.adaptive_loader_mode == "epoch_subset":
            distillation_enabled = True
            train_loader = curriculum.build_dataloader(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            pool_size = curriculum.current_pool_size()
            curriculum_finished = curriculum.curriculum_finished
        else:
            distillation_enabled = True
            train_loader = CurriculumBatchStream(
                curriculum=curriculum,
                batch_size=args.batch_size,
                num_batches=steps_per_epoch,
            )
            pool_size = curriculum.current_pool_size()
            curriculum_finished = curriculum.curriculum_finished
        log.train(len_dataset=len(train_loader))

        epoch_batches = 0
        for batch in train_loader:
            if curriculum is None:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                indices = None
            else:
                inputs, targets, indices = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                indices = indices.to(device)

            if args.optimizer == "sam":
                enable_running_stats(model)
                predictions = model(inputs)
                per_sample_loss = smooth_crossentropy(
                    predictions,
                    targets,
                    smoothing=args.label_smoothing,
                )
                first_loss = (
                    curriculum.curriculum_loss(per_sample_loss, predictions, indices)
                    if distillation_enabled
                    else per_sample_loss.mean()
                )
                first_loss.backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                second_predictions = model(inputs)
                second_per_sample_loss = smooth_crossentropy(
                    second_predictions,
                    targets,
                    smoothing=args.label_smoothing,
                )
                second_loss = (
                    curriculum.curriculum_loss(second_per_sample_loss, second_predictions, indices)
                    if distillation_enabled
                    else second_per_sample_loss.mean()
                )
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
                loss = first_loss
            else:
                predictions = model(inputs)
                per_sample_loss = smooth_crossentropy(
                    predictions,
                    targets,
                    smoothing=args.label_smoothing,
                )
                loss = (
                    curriculum.curriculum_loss(per_sample_loss, predictions, indices)
                    if distillation_enabled
                    else per_sample_loss.mean()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if curriculum is not None:
                curriculum.update_after_batch(model)

            cumulative_batches += 1
            epoch_batches += 1
            with torch.no_grad():
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.detach().cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))
        eval_loss_sum = 0.0
        eval_steps = 0
        eval_correct_sum = 0
        with torch.no_grad():
            for inputs, targets in dataset.test:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                eval_loss_sum += loss.sum().item()
                eval_steps += int(targets.numel())
                eval_correct_sum += int(correct.sum().item())

        epoch_val_accuracy = eval_correct_sum / eval_steps if eval_steps > 0 else float("nan")
        elapsed_seconds = time.perf_counter() - train_start_perf
        if curriculum is not None:
            pool_size = curriculum.current_pool_size()
            curriculum_finished = curriculum.curriculum_finished
        else:
            pool_size = ""
            curriculum_finished = ""
        val_curve.append(epoch_val_accuracy)

        with metrics_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["epoch", "cumulative_batches", "elapsed_seconds", "val_accuracy", "pool_size", "curriculum_finished"],
            )
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "cumulative_batches": cumulative_batches,
                    "elapsed_seconds": elapsed_seconds,
                    "val_accuracy": epoch_val_accuracy,
                    "pool_size": pool_size,
                    "curriculum_finished": curriculum_finished,
                }
            )

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_epoch = epoch + 1
            if args.save_student_checkpoint:
                torch.save(model.state_dict(), best_checkpoint_path)
                logger.info("Saved new best student checkpoint to %s", best_checkpoint_path)

        if curriculum is not None:
            logger.info(
                "Epoch %d/%d t: %.2fs  (T: %.2fs), epoch_batches=%d, pool_size=%d/%d, "
                "curriculum_finished=%s, lambda1=%.4f, val_accuracy=%.2f%%",
                epoch + 1,
                args.epochs,
                time.perf_counter() - epoch_start_perf,
                elapsed_seconds,
                epoch_batches,
                pool_size,
                len(dataset.train_dataset),
                curriculum.curriculum_finished,
                curriculum.config.lambda_kl,
                epoch_val_accuracy * 100,
            )
        else:
            logger.info(
                "Epoch %d/%d t: %.2fs  (T: %.2fs), epoch_batches=%d, val_accuracy=%.2f%%",
                epoch + 1,
                args.epochs,
                time.perf_counter() - epoch_start_perf,
                elapsed_seconds,
                epoch_batches,
                epoch_val_accuracy * 100,
            )

    log.flush()
    if best_epoch > 0:
        logger.info("Best validation accuracy: %.2f%% (epoch %d)", best_val_accuracy * 100, best_epoch)
    else:
        logger.warning("No epochs were run, so best validation accuracy is unavailable.")
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
