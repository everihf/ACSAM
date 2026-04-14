import argparse
import csv
import importlib.util

import torch
import logging
import time
from datetime import datetime
from copy import deepcopy
from zoneinfo import ZoneInfo

from model.wide_res_net import WideResNet
from model.cifar100_cnn import Cifar100CNN
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.adaptive_curriculum import AdaptiveCurriculum
from utility.fixed_curriculum import (
    FixedCurriculum,
    FixedCurriculumConfig,
    rank_samples_by_confidence,
    rank_samples_by_inception_svm,
    build_inception_svm_teacher_logits,
    balance_order_by_class,
)
from utility.teacher_model import pretrain_teacher_model
from utility.teacher_model import evaluate_accuracy
from utility.log import build_logger

from pathlib import Path
import sys

#否则无法正常导入sam这个库，因为sam.py和train.py不在同一个目录下，sam.py在根目录下，而train.py在example目录下。
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from sam import SAM

if importlib.util.find_spec("matplotlib.pyplot") is not None:
    import matplotlib.pyplot as plt
else:
    plt = None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_overridden_args(parser, args):
    overridden = {}
    for action in parser._actions:
        if not action.dest or action.dest == "help":
            continue
        if action.default is argparse.SUPPRESS:
            continue
        current_value = getattr(args, action.dest, None)
        if current_value != action.default:
            overridden[action.dest] = {
                "current": current_value,
                "default": action.default,
            }
    return overridden


def get_cli_provided_dests(parser, argv=None):
    """
    Return argparse destinations that were explicitly provided on CLI.

    This is different from get_overridden_args(): a user may pass an argument
    with the same value as its parser default (e.g. --learning_rate 0.1), and
    we still must treat it as user-specified.
    """
    tokens = sys.argv[1:] if argv is None else list(argv)
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    provided = set()
    for token in tokens:
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest and dest != "help":
            provided.add(dest)
    return provided


def apply_model_specific_safe_defaults(args, parser, cli_provided_dests=None):
    """
    Apply safer training defaults for cifar100_cnn without affecting WRN.

    Only parameters that still equal parser defaults are overridden, so explicit
    CLI values from users always take priority.
    """
    applied = {}
    if args.model != "cifar100_cnn":
        return applied

    cli_provided_dests = cli_provided_dests or set()
    safe_defaults = {
        "learning_rate": 0.01,
        "momentum": 0.0,
        "label_smoothing": 0.0,
    }
    for name, value in safe_defaults.items():
        if name in cli_provided_dests:
            continue
        if getattr(args, name) == parser.get_default(name):
            setattr(args, name, value)
            applied[name] = value
    return applied


def apply_dataset_specific_inception_svm_defaults(args, parser, cli_provided_dests=None):
    """
    Use faster SVM kernel defaults for large/common datasets.

    For cifar100/cifar10, if user did not explicitly pass --fixed_inception_svm_kernel,
    switch the default kernel from parser default to "linear".
    """
    applied = {}
    cli_provided_dests = cli_provided_dests or set()

    if "fixed_inception_svm_kernel" in cli_provided_dests:
        return applied

    if args.dataset not in {"cifar100", "cifar10"}:
        return applied

    if getattr(args, "fixed_inception_svm_kernel", None) == parser.get_default("fixed_inception_svm_kernel"):
        args.fixed_inception_svm_kernel = "linear"
        applied["fixed_inception_svm_kernel"] = "linear"

    return applied


def build_model(args, model_name, num_classes):
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
    raise ValueError(f"Unsupported model: {model_name}")


def resolve_curriculum_strategy(args):
    if args.curriculum_strategy is None:
        return "adaptive" if args.use_adaptive_curriculum else "none"
    return args.curriculum_strategy


def resolve_adaptive_curriculum_type(args):
    if args.adaptive_curriculum_type is not None:
        return str(args.adaptive_curriculum_type).lower()
    return "curriculum" if args.use_difficulty_sorting else "random"


class CurriculumBatchStream:
    """Epoch-sized iterable that samples batches from curriculum by global batch state."""

    def __init__(self, curriculum, batch_size: int, num_batches: int):
        self.curriculum = curriculum
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self.curriculum.sample_batch(self.batch_size)


def build_teacher_model(args, student_model, num_classes, device, logger):
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
        logger.info(
            "Using custom teacher WRN architecture: depth=%d, width_factor=%d, dropout=%.4f",
            teacher_depth,
            teacher_width_factor,
            teacher_dropout,
        )
    else:
        if teacher_model_name != "wrn" and use_custom_teacher_arch:
            logger.warning(
                "Ignoring --teacher_depth/--teacher_dropout/--teacher_width_factor because teacher_model=%s is not WRN.",
                teacher_model_name,
            )
        if args.teacher_model is None:
            teacher_model = deepcopy(student_model)
            logger.info("Teacher architecture defaults to a deepcopy of student model.")
        else:
            teacher_model = build_model(args, teacher_model_name, num_classes)
    teacher_model = teacher_model.to(device)
    logger.info("Teacher architecture: %s", teacher_model_name)
    return teacher_model


def prepare_teacher_model(args, student_model, dataset, device, logger, checkpoint_dir, run_name, log_prefix):
    teacher_log_path = Path(__file__).resolve().parent / f"{log_prefix}_teacher_seed{args.seed}.log"
    teacher_logger = build_logger("train.teacher", teacher_log_path)
    logger.info("Teacher pretraining log file: %s", teacher_log_path)

    teacher_model = build_teacher_model(args, student_model, len(dataset.classes), device, logger)
    if args.teacher_checkpoint:
        teacher_state = torch.load(args.teacher_checkpoint, map_location=device)
        teacher_model.load_state_dict(teacher_state)
        logger.info("Loaded teacher checkpoint from %s", args.teacher_checkpoint)
    else:
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


if __name__ == "__main__":
    #创建一个用来解析命令行参数的对象，让你的程序可以通过命令行接收输入
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=parse_bool, help="True if you want to use the Adaptive SAM.")#自适应SAM（ASAM）是SAM的一个变体，它在计算扰动时考虑了每个参数的绝对值。这意味着对于较大的参数，ASAM会施加更大的扰动，而对于较小的参数，扰动则较小。这种自适应机制可以帮助模型更有效地找到平坦的最小值，从而提高泛化性能。
    #数据集
    parser.add_argument(
        "--dataset",
        default="cifar100",
        type=str,
        choices=[
            "cifar10",
            "cifar100",
            "cifar100_aquatic_mammals",
            "cifar100_small_mammals",
            "cifar100_household_electrical_devices",
        ],
        help="Dataset to train on.",
    )
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size used in the training and validation loop.")#批量大小（batch size）
    parser.add_argument("--num_workers", default=2, type=int, help="Number of CPU threads for dataloaders.")
    #model
    parser.add_argument("--model", default="cifar100_cnn", type=str, choices=["wrn", "cifar100_cnn"], help="Model architecture to train.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--cifar100_activation", default="elu", type=str, choices=["elu", "relu", "gelu"], help="Activation used by cifar100_cnn model.")
    parser.add_argument("--cifar100_dropout1", default=0.25, type=float, help="Dropout after each convolutional stage in cifar100_cnn.")
    parser.add_argument("--cifar100_dropout2", default=0.5, type=float, help="Dropout before classifier head in cifar100_cnn.")
    parser.add_argument("--cifar100_batch_norm", default=False, type=parse_bool, help="Enable batch norm layers in cifar100_cnn.")
    parser.add_argument("--teacher_model", default=None, type=str, choices=["wrn", "cifar100_cnn"], help="Optional teacher architecture. If omitted, teacher reuses student's architecture.")
    parser.add_argument("--teacher_depth", default=None, type=int, help="Optional teacher WRN depth. Effective only when teacher model is WRN.")
    parser.add_argument("--teacher_dropout", default=None, type=float, help="Optional teacher WRN dropout. Effective only when teacher model is WRN.")
    parser.add_argument("--teacher_width_factor", default=None, type=int, help="Optional teacher WRN width factor. Effective only when teacher model is WRN.")
    parser.add_argument("--teacher_optimizer", default="sgd", type=str, choices=["sam", "sgd"], help="Optimizer used for teacher pretraining when no teacher checkpoint is provided.")
    #train
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sam", "sgd"], help="Training optimizer: 'sam' (default) or plain 'sgd' for control experiments.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")#v ← μ * v + g, w ← w - lr * v ;g是当前梯度，v是动量，μ是动量系数;当前更新 = 当前梯度 + 过去梯度的累积
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    # curriculum strategy
    parser.add_argument("--curriculum_strategy", default=None, type=str, choices=["none", "adaptive", "fixed", "self_paced"], help="Curriculum strategy. If omitted, falls back to --use_adaptive_curriculum for backward compatibility.")
    parser.add_argument("--use_adaptive_curriculum", default=True, type=parse_bool, help="Legacy flag. If --curriculum_strategy is omitted, True->adaptive, False->none.")
    parser.add_argument("--teacher_checkpoint", default="", type=str, help="Optional teacher checkpoint path. If empty, pretrain a teacher model first.")

    # adaptive curriculum params
    parser.add_argument("--pace_p", default=0.04, type=float, help="Initial curriculum ratio.")
    parser.add_argument("--pace_q", default=1.9, type=float, help="Curriculum growth base.")
    parser.add_argument("--pace_r", default=100, type=int, help="Curriculum growth interval in batches.")
    parser.add_argument("--inv", default=50, type=int, help="Difficulty update interval in batches.")
    parser.add_argument("--self_paced_inv", default=50, type=int, help="Difficulty update interval (in batches) used only by self-paced curriculum.")
    parser.add_argument("--alpha", default=-0.01, type=float, help="Difficulty EMA factor.")
    parser.add_argument(
        "--adaptive_teacher_source",
        default="inception_svm",
        type=str,
        choices=["inception_svm", "teacher_model"],
        help="Teacher source for adaptive curriculum distillation. "
             "'inception_svm' uses fixed-curriculum Inception+SVM pseudo teacher by default.",
    )
    parser.add_argument(
        "--use_difficulty_sorting",
        default=True,
        type=parse_bool,
        help="Legacy adaptive ordering flag. True->curriculum(easy->hard), False->random. "
             "Prefer using --adaptive_curriculum_type.",
    )
    parser.add_argument(
        "--adaptive_curriculum_type",
        default="curriculum",
        type=str,
        choices=["curriculum", "anti", "random", "self_paced"],
        help="Ordering style for adaptive curriculum candidate selection.",
    )
    parser.add_argument("--lambda1", default=0.01, type=float, help="Weight of teacher KL distillation term.")
    parser.add_argument("--lambda1_decay", default=None, type=float, help="Optional decay step for lambda1 at each inv interval.")
    parser.add_argument("--bottom_lambda1", default=0.1, type=float, help="Lower bound of lambda1 when decay is enabled.")
    parser.add_argument(
        "--distill_extra_epochs_after_curriculum",
        default=0,
        type=int,
        help="How many extra epochs to keep distillation after curriculum_finished=True. Set 0 to stop distillation immediately when curriculum finishes.",
    )
    # fixed curriculum params
    parser.add_argument("--fixed_curriculum_type", default="curriculum", type=str, choices=["curriculum", "anti", "random"], help="Ordering style for fixed curriculum.")
    parser.add_argument("--fixed_batch_increase", default=100, type=int, help="Every N batches, fixed curriculum increases available sample ratio.")
    parser.add_argument("--fixed_increase_amount", default=1.9, type=float, help="Exponential growth factor for fixed curriculum sampling ratio.")
    parser.add_argument("--fixed_starting_percent", default=100 / 2500, type=float, help="Initial visible data ratio for fixed curriculum.")
    parser.add_argument("--fixed_order_source", default="inception_svm", type=str, choices=["teacher", "student", "inception_svm"], help="Which source to use when computing fixed curriculum ordering scores.")
    parser.add_argument(
        "--fixed_balance_order",
        default=True,
        type=parse_bool,
        help="Whether to interleave ordering across classes to reduce early class imbalance "
             "(used by fixed curriculum and adaptive difficulty ordering).",
    )
    parser.add_argument("--fixed_inception_svm_kernel", default="rbf", type=str, choices=["rbf", "linear", "poly", "sigmoid"], help="SVM kernel for inception_svm fixed ordering.")
    parser.add_argument("--fixed_inception_svm_c", default=1.0, type=float, help="SVM C for inception_svm fixed ordering.")
    parser.add_argument("--fixed_inception_svm_gamma", default="scale", type=str, help="SVM gamma for inception_svm fixed ordering.")
    parser.add_argument(
        "--fixed_inception_svm_backend",
        default="auto",
        type=str,
        choices=["auto", "cuml", "sklearn"],
        help="SVM backend for inception_svm ordering. auto=prefer cuML(GPU), fallback sklearn(CPU).",
    )
    parser.add_argument("--fixed_inception_svm_cache", default=True, type=parse_bool, help="Whether to cache inception features and SVM scores for inception_svm ordering.")
    # metrics
    parser.add_argument("--metrics_dir", default="metrics", type=str, help="Directory (relative to example/) used to save validation metrics and plots.")
    parser.add_argument("--run_name", default="", type=str, help="可选运行名称，用于指标文件名。如果为空，则自动生成：时间戳+seed。")
    parser.add_argument("--method_name", default="", type=str, help="方法标签已保存到指标CSV文件中，以便后续多轮比较.")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="Directory (relative to example/) used to save model checkpoints.")
    parser.add_argument("--save_teacher_checkpoint", default=True, type=parse_bool, help="Whether to save teacher checkpoint when it is pretrained from scratch.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducible runs.")
    #解析参数
    args = parser.parse_args()
    cli_overridden_args = get_overridden_args(parser, args)
    cli_provided_dests = get_cli_provided_dests(parser)
    applied_safe_defaults = apply_model_specific_safe_defaults(args, parser, cli_provided_dests)
    applied_dataset_defaults = apply_dataset_specific_inception_svm_defaults(args, parser, cli_provided_dests)
    curriculum_strategy = resolve_curriculum_strategy(args)
    adaptive_curriculum_type = resolve_adaptive_curriculum_type(args)
    if curriculum_strategy == "adaptive" and adaptive_curriculum_type == "self_paced":
        curriculum_strategy = "self_paced"
    overridden_args = cli_overridden_args

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    initialize(args, seed=args.seed)
    train_start_time = datetime.now(ZoneInfo("Asia/Shanghai"))
    train_start_perf = time.perf_counter()
    log_prefix = train_start_time.strftime("%m-%d_%H-%M-%S")
    if curriculum_strategy == "none":
        student_log_filename = f"{log_prefix}_base_seed{args.seed}.log"
    elif curriculum_strategy == "adaptive":
        student_log_filename = f"{log_prefix}_student_seed{args.seed}.log"
    elif curriculum_strategy == "self_paced":
        student_log_filename = f"{log_prefix}_self-paced_seed{args.seed}.log"
    elif curriculum_strategy == "fixed":
        student_log_filename = f"{log_prefix}_fixed_seed{args.seed}.log"
    else:
        student_log_filename = f"{log_prefix}_student_seed{args.seed}.log"
    student_log_path = Path(__file__).resolve().parent / student_log_filename
    logger = build_logger("train.student", student_log_path)
    logger.info("Student training log file: %s", student_log_path)
    if applied_safe_defaults:
        logger.info(
            "Applied safe defaults for model=%s on non-overridden args: %s",
            args.model,
            ", ".join(f"{k}={v}" for k, v in applied_safe_defaults.items()),
        )
    if applied_dataset_defaults:
        logger.info(
            "Applied dataset-specific defaults for dataset=%s on non-overridden args: %s",
            args.dataset,
            ", ".join(f"{k}={v}" for k, v in applied_dataset_defaults.items()),
        )
    if args.curriculum_strategy is not None:
        legacy_expected = "adaptive" if args.use_adaptive_curriculum else "none"
        if legacy_expected != curriculum_strategy:
            logger.info(
                "Ignoring legacy --use_adaptive_curriculum=%s because --curriculum_strategy=%s was set explicitly.",
                args.use_adaptive_curriculum,
                curriculum_strategy,
            )
    logger.info(
        "Effective args: model=%s, optimizer=%s(adaptive=%s), curriculum_strategy=%s, teacher_optimizer=%s",
        args.model,
        args.optimizer,
        args.adaptive,
        curriculum_strategy,
        args.teacher_optimizer,
    )
    if overridden_args:
        logger.info("Detected non-default CLI args:")
        for name, value in sorted(overridden_args.items()):
            logger.info("  --%s: %s (default: %s)", name, value["current"], value["default"])
    if curriculum_strategy == "adaptive":
        logger.info(
            "Adaptive curriculum ordering type: %s",
            adaptive_curriculum_type,
        )
        if args.adaptive_curriculum_type is not None and "use_difficulty_sorting" in cli_provided_dests:
            legacy_expected_type = "curriculum" if args.use_difficulty_sorting else "random"
            if legacy_expected_type != adaptive_curriculum_type:
                logger.info(
                    "Ignoring legacy --use_difficulty_sorting=%s because --adaptive_curriculum_type=%s was set explicitly.",
                    args.use_difficulty_sorting,
                    adaptive_curriculum_type,
                )
        elif args.adaptive_curriculum_type is None:
            logger.info(
                "Adaptive ordering inferred from legacy --use_difficulty_sorting=%s.",
                args.use_difficulty_sorting,
            )
        logger.info(
            "Distillation extra epochs after curriculum finished: %d",
            max(0, args.distill_extra_epochs_after_curriculum),
        )
        logger.info(
            "Curriculum sample selection uses difficulty sorting: %s",
            args.use_difficulty_sorting,
        )
        logger.info(
            "Adaptive teacher source: %s",
            args.adaptive_teacher_source,
        )
        logger.info(
            "Adaptive class-balanced ordering (fixed_balance_order): %s",
            args.fixed_balance_order,
        )
        if args.adaptive_teacher_source == "inception_svm":
            logger.info(
                "Adaptive Inception+SVM config: backend=%s, kernel=%s, C=%.4f, gamma=%s, cache=%s",
                args.fixed_inception_svm_backend,
                args.fixed_inception_svm_kernel,
                args.fixed_inception_svm_c,
                args.fixed_inception_svm_gamma,
                args.fixed_inception_svm_cache,
            )
    elif curriculum_strategy == "self_paced":
        logger.info("Self-paced curriculum enabled (student-only difficulty, no teacher distillation).")
        logger.info(
            "Self-paced config: pace_p=%.4f, pace_q=%.4f, pace_r=%d, inv=%d, alpha=%.6f, balance_order=%s",
            args.pace_p,
            args.pace_q,
            args.pace_r,
            args.self_paced_inv,
            1.0,
            args.fixed_balance_order,
        )
    elif curriculum_strategy == "fixed":
        fixed_data_dir = ROOT_DIR / "data"
        inception_svm_cache_dir = fixed_data_dir / "inception_svm_cache"
        logger.info(
            "Fixed curriculum config: type=%s, source=%s, batch_increase=%d, increase_amount=%.4f, starting_percent=%.4f, balance_order=%s",
            args.fixed_curriculum_type,
            args.fixed_order_source,
            args.fixed_batch_increase,
            args.fixed_increase_amount,
            args.fixed_starting_percent,
            args.fixed_balance_order,
        )
        if args.fixed_order_source == "inception_svm":
            logger.info(
                "Inception+SVM ordering config: backend=%s, kernel=%s, C=%.4f, gamma=%s, cache=%s, data_dir=%s, cache_dir=%s",
                args.fixed_inception_svm_backend,
                args.fixed_inception_svm_kernel,
                args.fixed_inception_svm_c,
                args.fixed_inception_svm_gamma,
                args.fixed_inception_svm_cache,
                fixed_data_dir,
                inception_svm_cache_dir,
            )
    run_name = args.run_name or f"{train_start_time.strftime('%m-%d_%H-%M-%S')}_seed{args.seed}"
    checkpoint_dir = Path(__file__).resolve().parent / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = Cifar(args.batch_size, args.num_workers, dataset=args.dataset)
    train_log_each = 10 if curriculum_strategy == "fixed" else 50
    log = Log(log_each=train_log_each, logger=logger)
    model = build_model(args, args.model, len(dataset.classes)).to(device)
    logger.info("Student architecture: %s", args.model)

    curriculum = None
    if curriculum_strategy == "adaptive":
        adaptive_balance_enabled = bool(args.fixed_balance_order)
        if adaptive_balance_enabled and not hasattr(dataset.train.dataset, "targets"):
            adaptive_balance_enabled = False
            logger.warning(
                "Adaptive class-balanced ordering requested but train dataset has no 'targets'; disabling it."
            )
        teacher_model = None
        teacher_logits_by_index = None
        if args.adaptive_teacher_source == "inception_svm":
            teacher_logits_by_index = build_inception_svm_teacher_logits(
                train_dataset=dataset.train.dataset,
                dataset_name=args.dataset,
                num_classes=len(dataset.classes),
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                cache_dir=ROOT_DIR / "data" / "inception_svm_cache",
                svm_kernel=args.fixed_inception_svm_kernel,
                svm_c=args.fixed_inception_svm_c,
                svm_gamma=args.fixed_inception_svm_gamma,
                use_cache=args.fixed_inception_svm_cache,
                svm_backend=args.fixed_inception_svm_backend,
            )
            logger.info("Using Inception+SVM pseudo teacher logits for adaptive curriculum distillation.")
        else:
            teacher_model = prepare_teacher_model(
                args=args,
                student_model=model,
                dataset=dataset,
                device=device,
                logger=logger,
                checkpoint_dir=checkpoint_dir,
                run_name=run_name,
                log_prefix=log_prefix,
            )

        curriculum = AdaptiveCurriculum(
            train_dataset=dataset.train.dataset,
            teacher_model=teacher_model,
            teacher_logits_by_index=teacher_logits_by_index,
            device=device,
            num_classes=len(dataset.classes),
            pace_p=args.pace_p,
            pace_q=args.pace_q,
            pace_r=args.pace_r,
            inv=args.inv,
            alpha=args.alpha,
            lambda1=args.lambda1,
            lambda1_decay=args.lambda1_decay,
            bottom_lambda1=args.bottom_lambda1,
            curriculum_type=adaptive_curriculum_type,
            use_difficulty_sorting=args.use_difficulty_sorting,
            use_balance_order=adaptive_balance_enabled,
        )
        curriculum.initialize(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            model=model,
        )
    elif curriculum_strategy == "self_paced":
        adaptive_balance_enabled = bool(args.fixed_balance_order)
        if adaptive_balance_enabled and not hasattr(dataset.train.dataset, "targets"):
            adaptive_balance_enabled = False
            logger.warning(
                "Self-paced class-balanced ordering requested but train dataset has no 'targets'; disabling it."
            )
        curriculum = AdaptiveCurriculum(
            train_dataset=dataset.train.dataset,
            teacher_model=None,
            teacher_logits_by_index=None,
            device=device,
            num_classes=len(dataset.classes),
            pace_p=args.pace_p,
            pace_q=args.pace_q,
            pace_r=args.pace_r,
            inv=args.self_paced_inv,
            alpha=1.0,
            lambda1=0.0,
            lambda1_decay=None,
            bottom_lambda1=0.0,
            curriculum_type="self_paced",
            use_difficulty_sorting=True,
            use_balance_order=adaptive_balance_enabled,
            student_difficulty_only=True,
        )
        curriculum.initialize(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            model=model,
        )
    elif curriculum_strategy == "fixed":
        if args.fixed_order_source == "teacher":
            order_model = prepare_teacher_model(
                args=args,
                student_model=model,
                dataset=dataset,
                device=device,
                logger=logger,
                checkpoint_dir=checkpoint_dir,
                run_name=run_name,
                log_prefix=log_prefix,
            )
        else:
            if args.fixed_order_source == "student":
                order_model = deepcopy(model).to(device)
                logger.info("Using student model snapshot to compute fixed curriculum ordering.")
                order_model.eval()
                ordered_indices = rank_samples_by_confidence(
                    model=order_model,
                    train_dataset=dataset.train.dataset,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
            else:
                ordered_indices = rank_samples_by_inception_svm(
                    train_dataset=dataset.train.dataset,
                    dataset_name=args.dataset,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    cache_dir=ROOT_DIR / "data" / "inception_svm_cache",
                    svm_kernel=args.fixed_inception_svm_kernel,
                    svm_c=args.fixed_inception_svm_c,
                    svm_gamma=args.fixed_inception_svm_gamma,
                    use_cache=args.fixed_inception_svm_cache,
                    svm_backend=args.fixed_inception_svm_backend,
                )
                logger.info("Using Inception+SVM transfer ranking for fixed curriculum ordering.")
        if args.fixed_order_source == "teacher":
            order_model.eval()
            ordered_indices = rank_samples_by_confidence(
                model=order_model,
                train_dataset=dataset.train.dataset,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        if args.fixed_balance_order:
            train_targets = getattr(dataset.train.dataset, "targets", None)
            if train_targets is not None:
                ordered_indices = balance_order_by_class(
                    ordered_indices=ordered_indices,
                    labels=list(train_targets),
                    num_classes=len(dataset.classes),
                )
            else:
                logger.warning("Train dataset has no 'targets' attribute; skip fixed curriculum class balancing.")

        curriculum = FixedCurriculum(
            train_dataset=dataset.train.dataset,
            ordered_indices=ordered_indices,
            config=FixedCurriculumConfig(
                batch_increase=args.fixed_batch_increase,
                increase_amount=args.fixed_increase_amount,
                starting_percent=args.fixed_starting_percent,
                curriculum_type=args.fixed_curriculum_type,
            ),
        )
        curriculum.initialize(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    if args.optimizer == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
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
    cumulative_batches = 0
    val_curve = []
    steps_per_epoch = len(dataset.train)

    metrics_dir = Path(__file__).resolve().parent / args.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    default_method_name = args.optimizer
    if args.optimizer == "sam" and args.adaptive:
        default_method_name = "asam"
    if curriculum_strategy == "adaptive":
        default_method_name = f"{default_method_name}+adaptive_curriculum-{adaptive_curriculum_type}"
        if args.teacher_optimizer == "sgd":
            default_method_name = f"{default_method_name}-{args.teacher_optimizer}"
        elif args.teacher_optimizer == "sam":
            default_method_name = f"{default_method_name}-{args.teacher_optimizer}"
    elif curriculum_strategy == "self_paced":
        default_method_name = f"{default_method_name}+self_paced_curriculum"
    elif curriculum_strategy == "fixed":
        default_method_name = (
            f"{default_method_name}+fixed_curriculum-{args.fixed_curriculum_type}-{args.fixed_order_source}"
        )
    method_name = args.method_name or default_method_name#如果 args.method_name 有值 → 用它  ,否则 → 用 default_method_name
    csv_path = metrics_dir / f"{run_name}_val_curve.csv"
    plot_path = metrics_dir / f"{run_name}_val_curve.png"
    best_val_accuracy = float("-inf")
    best_epoch = -1
    curriculum_finished_epoch = None

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["run_name", "method", "epoch", "cumulative_batches", "elapsed_seconds", "val_accuracy"],
        )
        writer.writeheader()

    train_start_perf = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        ###模型训练
        model.train()
        train_loader = dataset.train
        if curriculum_strategy in {"fixed", "adaptive", "self_paced"} and curriculum is not None:
            train_loader = CurriculumBatchStream(
                curriculum=curriculum,
                batch_size=args.batch_size,
                num_batches=steps_per_epoch,
            )
            if curriculum_strategy == "fixed":
                log.train(
                    len_dataset=len(train_loader),
                    reset_step=False,
                    reset_last_steps=False,
                )
            else:
                log.train(len_dataset=len(train_loader))
        else:
            if curriculum is not None:
                if curriculum.curriculum_finished:
                    if curriculum_finished_epoch is None:
                        curriculum_finished_epoch = epoch + 1
                    train_loader = curriculum.build_full_dataloader(
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                else:
                    train_loader = curriculum.build_dataloader(
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                    if curriculum.curriculum_finished and curriculum_finished_epoch is None:
                        curriculum_finished_epoch = epoch + 1
            log.train(len_dataset=len(train_loader))
        distillation_enabled = False
        if curriculum_strategy == "adaptive" and curriculum is not None:
            extra_epochs = max(0, args.distill_extra_epochs_after_curriculum)
            if curriculum.curriculum_finished and curriculum_finished_epoch is None:
                curriculum_finished_epoch = epoch + 1
            if not curriculum.curriculum_finished:
                distillation_enabled = True
            elif curriculum_finished_epoch is not None:
                distillation_enabled = (epoch + 1) < (curriculum_finished_epoch + extra_epochs)

        epoch_batches = 0
        for batch in train_loader:
            if curriculum is not None:
                inputs, targets, indices = batch
                inputs, targets, indices = inputs.to(device), targets.to(device), indices.to(device)
            else:
                inputs, targets = (b.to(device) for b in batch)
                indices = None

            if args.optimizer == "sam":
                ### first forward-backward step
                #根据当前梯度，构造一个扰动 e(w) ,w→w+e(w)
                enable_running_stats(model)
                #把模型里 BatchNorm 层的 momentum 恢复成原来的值，让 BN 继续正常更新 running mean / running var
                predictions = model(inputs)
                per_sample_loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)#标签平滑（Label Smoothing）版交叉熵
                if curriculum_strategy == "adaptive" and curriculum is not None and distillation_enabled:
                    # 课程学习loss：监督损失 + 蒸馏 KL
                    first_loss = curriculum.curriculum_loss(per_sample_loss, predictions, indices)
                else:
                    first_loss = per_sample_loss.mean()
                first_loss.backward()
                optimizer.first_step(zero_grad=True)

                ### second forward-backward step
                #在这个扰动后的参数点w→w+e(w)上重新算梯度，再真正更新原始参数。
                disable_running_stats(model)
                second_predictions = model(inputs)
                second_per_sample_loss = smooth_crossentropy(second_predictions, targets, smoothing=args.label_smoothing)
                if curriculum_strategy == "adaptive" and curriculum is not None and distillation_enabled:
                    # second step 保持同样的课程loss，确保 SAM 两步一致。
                    second_loss = curriculum.curriculum_loss(second_per_sample_loss, second_predictions, indices)
                else:
                    second_loss = second_per_sample_loss.mean()
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
                loss = first_loss
            else: # sgd 
                predictions = model(inputs)
                per_sample_loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)#标签平滑（Label Smoothing）版交叉熵
                if curriculum_strategy == "adaptive" and curriculum is not None and distillation_enabled:
                    loss = curriculum.curriculum_loss(per_sample_loss, predictions, indices)
                else:
                    loss = per_sample_loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if curriculum is not None:
                # 每个batch后更新课程状态（全局batch计数、难度刷新、lambda1衰减）。
                curriculum.update_after_batch(
                    model=model,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                if (
                    curriculum_strategy in {"adaptive", "self_paced"}
                    and curriculum_finished_epoch is None
                    and curriculum.curriculum_finished
                ):
                    curriculum_finished_epoch = epoch + 1

            cumulative_batches += 1
            epoch_batches += 1
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        ###模型评估
        model.eval()
        if curriculum_strategy == "fixed":
            log.eval(len_dataset=len(dataset.test), reset_step=False)
        else:
            log.eval(len_dataset=len(dataset.test))
        eval_loss_sum = 0.0
        eval_steps = 0
        eval_correct_sum = 0

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
                eval_loss_sum += loss.sum().item()
                eval_steps += int(targets.numel())
                eval_correct_sum += int(correct.sum().item())

        epoch_val_loss = eval_loss_sum / eval_steps if eval_steps > 0 else float("nan")
        epoch_val_accuracy = eval_correct_sum / eval_steps if eval_steps > 0 else float("nan")
        elapsed_since_start_seconds = time.perf_counter() - train_start_perf
        val_curve.append(
            {
                "epoch": epoch + 1,
                "cumulative_batches": cumulative_batches,
                "elapsed_seconds": elapsed_since_start_seconds,
                "val_accuracy": epoch_val_accuracy,
            }
        )
        with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["run_name", "method", "epoch", "cumulative_batches", "elapsed_seconds", "val_accuracy"],
            )
            writer.writerow(
                {
                    "run_name": run_name,
                    "method": method_name,
                    "epoch": epoch + 1,
                    "cumulative_batches": cumulative_batches,
                    "elapsed_seconds": elapsed_since_start_seconds,
                    "val_accuracy": epoch_val_accuracy,
                }
            )


        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            best_epoch = epoch + 1
            if best_val_accuracy > 0.95:
                logger.info(
                    "New best validation accuracy at epoch %d: %.2f%%",
                    best_epoch,
                    best_val_accuracy * 100,
                )

        epoch_duration_seconds = time.perf_counter() - epoch_start_time
        logger.info(
            "Epoch %d/%d t: %.2fs  (T: %.2fs), "
            "epoch_batches=%d (cumulative_batches=%d), val_accuracy=%.2f%%, val_loss=%.4f",
            epoch + 1,
            args.epochs,
            epoch_duration_seconds,
            elapsed_since_start_seconds,
            epoch_batches,
            cumulative_batches,
            epoch_val_accuracy * 100,   # ⭐ 这里乘100
            epoch_val_loss,
        )

    log.flush()#打印/冲洗 log
    logger.info("Saved validation curve data to %s", csv_path)

    total_training_seconds = (
        datetime.now(ZoneInfo("Asia/Shanghai")) - train_start_time
    ).total_seconds()
    logger.info("Training finished in %.2f seconds", total_training_seconds)
    if best_epoch > 0:
        logger.info(
            "Best validation accuracy: %.2f%% (epoch %d)",
            best_val_accuracy * 100,
            best_epoch,
        )
    else:
        logger.warning(
            "Best validation accuracy is unavailable (epochs=%d, eval_steps may be 0).",
            args.epochs,
        )

    if plt is not None and len(val_curve) > 0:
        try:
            x = [point["cumulative_batches"] for point in val_curve]
            y = [point["val_accuracy"] for point in val_curve]
            plt.figure(figsize=(8, 5))
            plt.plot(x, y, marker="o", linewidth=1.5)
            plt.xlabel("Cumulative Training Batches")
            plt.ylabel("Validation Accuracy")
            plt.title(f"Validation Accuracy Curve ({method_name})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
            logger.info("Saved validation curve plot to %s", plot_path)
        except Exception:
            logger.exception("Failed to save validation curve plot to %s", plot_path)
    else:
        logger.warning("matplotlib is not available; skipped saving validation curve plot.")

