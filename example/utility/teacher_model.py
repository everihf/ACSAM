import time
import sys
from copy import deepcopy
from pathlib import Path
import torch
from utility.step_lr import StepLR
from utility.log import Log
# Ensure the repository root is importable when this module is loaded from example/train.py.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from sam import SAM
from utility.bypass_bn import enable_running_stats, disable_running_stats
from model.smooth_cross_entropy import smooth_crossentropy

def pretrain_teacher_model(
    teacher_model,
    train_loader,
    test_loader,
    args,
    device,
    logger,
    best_checkpoint_path=None,
):
    """Train a teacher from scratch when no checkpoint is provided."""
    logger.info(
        "No teacher checkpoint provided. Pretraining teacher for %d epochs with %s optimizer before adaptive curriculum.",
        args.epochs,
        args.teacher_optimizer.upper(),
    )
    if args.teacher_optimizer == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            teacher_model.parameters(),
            base_optimizer,
            rho=args.rho,
            adaptive=args.adaptive,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            teacher_model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
    teacher_log = Log(log_each=50, logger=logger)
    teacher_best_val_accuracy = float("-inf")
    teacher_best_epoch = -1
    teacher_best_state_dict = None
    if best_checkpoint_path is not None:
        best_checkpoint_path = Path(best_checkpoint_path)
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pretrain_start_perf = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        teacher_model.train()
        teacher_log.train(len_dataset=len(train_loader))
        epoch_batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if args.teacher_optimizer == "sam":
                enable_running_stats(teacher_model)
                predictions = teacher_model(inputs)
                first_loss = smooth_crossentropy(
                    predictions,
                    targets,
                    smoothing=args.label_smoothing,
                ).mean()
                first_loss.backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(teacher_model)
                second_predictions = teacher_model(inputs)
                second_loss = smooth_crossentropy(
                    second_predictions,
                    targets,
                    smoothing=args.label_smoothing,
                ).mean()
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
                loss = first_loss
            else:
                predictions = teacher_model(inputs)
                loss = smooth_crossentropy(
                    predictions,
                    targets,
                    smoothing=args.label_smoothing,
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                teacher_log(teacher_model, loss.detach().cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
            epoch_batches += 1

        teacher_model.eval()
        teacher_log.eval(len_dataset=len(test_loader))
        eval_loss_sum = 0.0
        eval_steps = 0
        eval_correct_sum = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = teacher_model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                teacher_log(teacher_model, loss.cpu(), correct.cpu())
                eval_loss_sum += loss.sum().item()
                eval_steps += int(targets.numel())
                eval_correct_sum += int(correct.sum().item())

        epoch_val_loss = eval_loss_sum / eval_steps if eval_steps > 0 else float("nan")
        epoch_val_accuracy = eval_correct_sum / eval_steps if eval_steps > 0 else float("nan")
        if epoch_val_accuracy > teacher_best_val_accuracy:
            teacher_best_val_accuracy = epoch_val_accuracy
            teacher_best_epoch = epoch + 1
            if isinstance(teacher_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                teacher_best_state_dict = deepcopy(teacher_model.module.state_dict())
            else:
                teacher_best_state_dict = deepcopy(teacher_model.state_dict())#暂时保存最佳模型的权重参数到内存，最好训练完成再写入磁盘
            if epoch_val_accuracy>0.95:
                logger.info(
                "Teacher pretrain new best validation accuracy at epoch %d: %.2f%%",
                teacher_best_epoch,
                teacher_best_val_accuracy * 100,
            )
            

        epoch_duration_seconds = time.perf_counter() - epoch_start_time
        elapsed_since_start_seconds = time.perf_counter() - pretrain_start_perf
        logger.info(
            "Teacher pretrain epoch %d/%d t: %.2fs  (T: %.2fs), "
            "epoch_batches=%d, val_accuracy=%.2f%%, val_loss=%.4f",
            epoch + 1,
            args.epochs,
            epoch_duration_seconds,
            elapsed_since_start_seconds,
            epoch_batches,
            epoch_val_accuracy * 100,
            epoch_val_loss,
        )

    teacher_log.flush()
    if teacher_best_epoch > 0:
        logger.info(
            "Teacher pretrain best validation accuracy: %.2f%% (epoch %d)",
            teacher_best_val_accuracy * 100,
            teacher_best_epoch,
        )
        if teacher_best_state_dict is not None:
            if best_checkpoint_path is not None:
                torch.save(teacher_best_state_dict, best_checkpoint_path)
                logger.info("Saved teacher best checkpoint to %s", best_checkpoint_path)
            if isinstance(teacher_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                teacher_model.module.load_state_dict(teacher_best_state_dict)
            else:
                teacher_model.load_state_dict(teacher_best_state_dict)#有这步加载：return 的是 best；没这步加载：return 的是 last。
            logger.info("Loaded teacher best checkpoint into model (epoch %d)", teacher_best_epoch)
    teacher_model.eval()
    return teacher_model


def evaluate_accuracy(model: torch.nn.Module, data_loader, device: torch.device) -> float:
    """Evaluate top-1 accuracy on a dataloader."""
    model.eval()
    correct_sum = 0
    sample_sum = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            correct = torch.argmax(predictions, 1) == targets
            correct_sum += int(correct.sum().item())
            sample_sum += int(targets.numel())
    return (correct_sum / sample_sum) if sample_sum > 0 else float("nan")
