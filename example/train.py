import argparse
import torch
import logging
from datetime import datetime

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

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
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of CPU threads for dataloaders.")
    #model
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
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
    #解析参数
    args = parser.parse_args()

    train_start_time = datetime.now()
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
        ###模型训练
        model.train()
        log.train(len_dataset=len(dataset.train))#载入训练集长度

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            if args.optimizer == "sam":
                ### first forward-backward step
                #根据当前梯度，构造一个扰动 e(w) ,w→w+e(w)
                enable_running_stats(model)
                #把模型里 BatchNorm 层的 momentum 恢复成原来的值，让 BN 继续正常更新 running mean / running var
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)#标签平滑（Label Smoothing）版交叉熵
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                ### second forward-backward step
                #在这个扰动后的参数点w→w+e(w)上重新算梯度，再真正更新原始参数。
                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)
            else:
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)#标签平滑（Label Smoothing）版交叉熵
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

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

    log.flush()#打印/冲洗 log
