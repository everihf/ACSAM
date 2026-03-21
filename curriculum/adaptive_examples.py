import argparse

from curriculum.algorithms import \
    BaseTrainer, AdaptiveTrainer

#test yes!
#test yes!

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--net', type=str, default='resnet')
    #curriculum\backbones\__init__.py中定义的网络模型名称！，resnet--ResNet18!
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_classes', type=int, default=10)
    
    parser.add_argument('--pace_p', type=float, default=0.04)
    parser.add_argument('--pace_q', type=float, default=1.1)
    parser.add_argument('--pace_r', type=int, default=100)
    parser.add_argument('--inv', type=int, default=50)
    #
    parser.add_argument('--alpha', type=float, default=-0.01)
    #
    parser.add_argument('--lambda1', type=float, default=0.01)
    #之前默认是0.1，但是ACL论文推荐0.01
    parser.add_argument('--lambda1_decay', type=float, default=None)
    parser.add_argument('--bottom_lambda1', type=float, default=0.1)
    parser.add_argument('--teacher_dir', type=str, default=None)#例如'runs/teacher_model'
    #添加教师模型！
    args = parser.parse_args()

    pretrainer = BaseTrainer(
        data_name=args.data,
        net_name=args.net,
        device_name=args.device,
        num_epochs=args.epochs,
        random_seed=42,
    )
    
    #若没有教师模型，就先训练一个教师模型！
    if args.teacher_dir is None:
        pretrainer.fit()
    pretrainer.evaluate(args.teacher_dir)#评估教师模型的性能

    teacher_net = pretrainer.export(args.teacher_dir)
    #将该文件夹下的预训练模型导出为teacher_net

    trainer = AdaptiveTrainer(
        data_name=args.data,
        net_name=args.net,
        device_name=args.device,
        num_epochs=args.epochs,
        random_seed=args.seed,
        num_classes=args.num_classes,
        pace_p=args.pace_p,
        pace_q=args.pace_q,
        pace_r=args.pace_r,
        inv=args.inv,
        alpha=args.alpha,
        lambda1=args.lambda1,
        lambda1_decay=args.lambda1_decay,
        bottom_lambda1=args.bottom_lambda1,
        pretrained_net=teacher_net,
        #这里不要修改，在前面参数设置添加教师模型
    )
    trainer.fit()
    trainer.evaluate()


if __name__ == "__main__":
    main()