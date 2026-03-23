import argparse
import torch
import logging
import time
from datetime import datetime
from copy import deepcopy

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.adaptive_curriculum import AdaptiveCurriculum

from pathlib import Path
import sys

#否则无法正常导入sam这个库，因为sam.py和train.py不在同一个目录下，sam.py在根目录下，而train.py在example目录下。
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from sam import SAM


if __name__ == "__main__":
    #创建一个用来解析命令行参数的对象，让你的程序可以通过命令行接收输入
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")#自适应SAM（ASAM）是SAM的一个变体，它在计算扰动时考虑了每个参数的绝对值。这意味着对于较大的参数，ASAM会施加更大的扰动，而对于较小的参数，扰动则较小。这种自适应机制可以帮助模型更有效地找到平坦的最小值，从而提高泛化性能。
    #数据集
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100"], help="Dataset to train on.")
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size used in the training and validation loop.")#批量大小（batch size）
    parser.add_argument("--num_workers", default=2, type=int, help="Number of CPU threads for dataloaders.")
    #model
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")#WRN-16-8 中的 16 就是 depth，表示网络的深度，即层数。WRN-16-8 是 Wide ResNet 的一个变体，其中 16 表示网络的深度，8 表示宽度因子（width factor）。在 WRN 中，depth 通常是 6n+4 的形式，其中 n 是一个整数，表示每个阶段（stage）中 BasicUnit 的数量。因此，WRN-16-8 中的 depth=16 意味着每个阶段有 2 个 BasicUnit（因为 (16-4)/6=2），总共有 3 个阶段（stage），加上初始卷积层和最后的全连接层，总共是 16 层。
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")#比普通ResNet宽多少倍
    #train
    parser.add_argument("--optimizer", default="sam", type=str, choices=["sam", "sgd"], help="Training optimizer: 'sam' (default) or plain 'sgd' for control experiments.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    # adaptive curriculum
    parser.add_argument("--use_adaptive_curriculum", default=False, type=bool, help="Enable adaptive curriculum + distillation while keeping SAM/ASAM optimizer.")
    parser.add_argument("--teacher_checkpoint", default="", type=str, help="Optional teacher checkpoint path. If empty, use the model initialization snapshot as teacher.")
    parser.add_argument("--pace_p", default=0.04, type=float, help="Initial curriculum ratio.")
    parser.add_argument("--pace_q", default=1.1, type=float, help="Curriculum growth base.")
    parser.add_argument("--pace_r", default=100, type=int, help="Curriculum growth interval in batches.")
    parser.add_argument("--inv", default=50, type=int, help="Difficulty update interval in batches.")
    parser.add_argument("--alpha", default=-0.01, type=float, help="Difficulty EMA factor.")
    parser.add_argument("--lambda1", default=0.01, type=float, help="Weight of teacher KL distillation term.")
    parser.add_argument("--lambda1_decay", default=None, type=float, help="Optional decay step for lambda1 at each inv interval.")
    parser.add_argument("--bottom_lambda1", default=0.1, type=float, help="Lower bound of lambda1 when decay is enabled.")
    #解析参数
    args = parser.parse_args()

    train_start_time = datetime.now()
    train_start_perf = time.perf_counter()
    log_filename = train_start_time.strftime("%m-%d_%H-%M.log")
    log_path = Path(__file__).resolve().parent / log_filename
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("train")

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.num_workers, dataset=args.dataset)
    log = Log(log_each=10, logger=logger)#每 10 个 step 打印一次训练中间结果
    model = WideResNet(
        args.depth,
        args.width_factor,
        args.dropout,
        in_channels=3,
        labels=len(dataset.classes),
    ).to(device)
    #WideResnet充当model

    curriculum = None
    if args.use_adaptive_curriculum:
        # 与原 SAM 代码保持一致：学生模型仍然是同一个 WideResNet，
        # 课程学习只是在数据采样和loss上做附加，不改模型定义。
        teacher_model = deepcopy(model)
        if args.teacher_checkpoint:
            teacher_state = torch.load(args.teacher_checkpoint, map_location=device)
            teacher_model.load_state_dict(teacher_state)
            logger.info("Loaded teacher checkpoint from %s", args.teacher_checkpoint)
        else:
            logger.warning("No teacher checkpoint provided, using model initialization snapshot as teacher.")

        curriculum = AdaptiveCurriculum(
            # 数据仍使用原始 CIFAR 训练集；内部只会包装 index 供课程学习使用。
            train_dataset=dataset.train.dataset,
            teacher_model=teacher_model,
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

    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        ###模型训练
        model.train()
        train_loader = dataset.train
        if curriculum is not None:
            # 每个epoch按当前 difficulty/pace 重新构造课程子集。
            train_loader = curriculum.build_dataloader(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        log.train(len_dataset=len(train_loader))#载入训练集长度

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
                if curriculum is not None:
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
                if curriculum is not None:
                    # second step 保持同样的课程loss，确保 SAM 两步一致。
                    second_loss = curriculum.curriculum_loss(second_per_sample_loss, second_predictions, indices)
                else:
                    second_loss = second_per_sample_loss.mean()
                second_loss.backward()
                optimizer.second_step(zero_grad=True)
                loss = first_loss
            else:
                predictions = model(inputs)
                per_sample_loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)#标签平滑（Label Smoothing）版交叉熵
                if curriculum is not None:
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

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        ###模型评估
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

        epoch_duration_seconds = time.perf_counter() - epoch_start_time
        elapsed_since_start_seconds = time.perf_counter() - train_start_perf
        logger.info(
            "Epoch %d/%d finished in %.2f seconds (total elapsed: %.2f seconds)",
            epoch + 1,
            args.epochs,
            epoch_duration_seconds,
            elapsed_since_start_seconds,
        )

    log.flush()#打印/冲洗 log
    total_training_seconds = (datetime.now() - train_start_time).total_seconds()
    logger.info("Training finished in %.2f seconds", total_training_seconds)
