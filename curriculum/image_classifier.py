import os
import time
import torch
from torch.utils.data import DataLoader

from ..datasets import get_dataset_with_noise
from ..backbones import get_net
from ..utils import get_logger, set_random


class ImageClassifier():
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 algorithm_name, data_prepare, model_prepare, data_curriculum,
                 model_curriculum, loss_curriculum):
        self.random_seed = random_seed
        set_random(self.random_seed)

        self.algorithm_name = algorithm_name
        self.data_prepare = data_prepare
        self.model_prepare = model_prepare
        self.data_curriculum = data_curriculum
        self.model_curriculum = model_curriculum
        self.loss_curriculum = loss_curriculum

        self._init_dataloader(data_name)
        self._init_model(data_name, net_name, device_name, num_epochs)
        self._init_logger(algorithm_name, data_name, net_name, num_epochs, random_seed)

    def _init_dataloader(self, data_name):
        #数据集：训练集，验证集，测试集
        train_dataset, valid_dataset, test_dataset = \
            get_dataset_with_noise('./data', data_name)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=100, shuffle=True, num_workers=2, pin_memory=True)

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

        self.data_prepare(self.train_loader)

    def _init_model(self, data_name, net_name, device_name, num_epochs):
        self.net = get_net(net_name, data_name)
        self.device = torch.device(device_name \
                                       if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.SGD(
            self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        #学习率，动量，权重衰减
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6)
        #余弦退火学习率（Cosine Annealing）/T_max=self.epochs:退火周期，这里是整个训练过程下降一次/eta_min最小学习率：0.000001，避免学习率降到 0。

        self.model_prepare(
            self.net, self.device, self.epochs,
            self.criterion, self.optimizer, self.lr_scheduler)

    def _init_logger(self, algorithm_name, data_name,
                     net_name, num_epochs, random_seed):
        log_info = '%s-%s-%s-%d-%d-%s' % (
            algorithm_name, data_name, net_name, num_epochs, random_seed,
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        self.log_dir = os.path.join('./runs', log_info)
        if not os.path.exists('./runs'): os.mkdir('./runs')
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        else:
            print('The directory %s has already existed.' % (self.log_dir))

        self.log_interval = 1
        self.batch_log_interval = 50
        #每训练50个batch记录一次日志，或者每个epoch结束记录一次日志
        self.logger = get_logger(os.path.join(self.log_dir, 'train.log'), log_info)

    def _train(self):
        best_acc = 0.0
        t0= time.time()
        for epoch in range(self.epochs):
            t = time.time()
            total = 0
            correct = 0
            train_loss = 0.0

            net = self.model_curriculum(self.net)  # curriculum part
            net.train()

            steps_done_epoch = 0#用来算每个epoch的平均损失
            if self.algorithm_name == 'adaptive':
                adaptive_algo = getattr(self.data_curriculum, '__self__',None )#default=None

                # 课程已经扩展到全数据集后，退化为普通for循环，避免每个step重复重建loader。
                if getattr(adaptive_algo, 'curriculum_finished', False):#default=False
                    # 课程结束后直接按常规方式遍历训练集，不再调用adaptive的数据抓取逻辑。
                    # 注意：adaptive 的 loss 依赖样本原始索引（data[2]），因此必须使用 CLDataset。
                    if adaptive_algo is None or not hasattr(adaptive_algo, 'dataset'):
                        raise RuntimeError('Adaptive curriculum requires CLDataset with sample indices.')
                    loader = DataLoader(
                        adaptive_algo.dataset,
                        batch_size=self.train_loader.batch_size,
                        shuffle=True,
                        #注意num_workers会影响训练时间（载入数据）
                        num_workers=self.train_loader.num_workers,
                        pin_memory=self.train_loader.pin_memory,
                    )
                    num_steps = len(loader)
                    for step, data in enumerate(loader):
                        inputs = data[0].to(self.device)
                        labels = data[1].to(self.device)
                        indices = data[2].to(self.device)

                        self.optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = self.loss_curriculum(  # curriculum part
                            self.criterion, outputs, labels, indices)
                        loss.backward()
                        self.optimizer.step()

                        if adaptive_algo is not None and hasattr(adaptive_algo, 'update_after_curriculum_finished_step'):
                            adaptive_algo.update_after_curriculum_finished_step()

                        train_loss += loss.item()
                        _, predicted = outputs.max(dim=1)
                        correct += predicted.eq(labels).sum().item()
                        total += labels.shape[0]

                        steps_done_epoch = step + 1

                        if (step + 1) % self.batch_log_interval == 0 or (step + 1) == num_steps:
                        #因为for循环的step是从0开始的，所以要加1才能正确记录日志
                            steps_done = step + 1#用来算每个inv（50个batch)的平均损失
                            self.logger.info(
                                '[%3d]  Step %4d/%4d  Train Acc = %.4f  Loss = %.4f'
                                % (epoch + 1, steps_done, num_steps,
                                   correct / total, train_loss / steps_done))
                            
                #adaptive算法：训练没扩展到全数据集的情况。调用adaptive的数据抓取逻辑，每个batch的训练集大小不一样。
                else:
                    num_steps = 0 #训练集的step(batch)数量，因为adaptive算法每个batch的训练集大小不一样
                    step = 0      #已经训练的step（batch)数量
                    while True:#因为adaptive下每个batch的训练集大小不一样，所以不能直接用for循环迭代训练集，而是用while循环，每个batch结束后重新计算训练集大小，并判断是否结束该epoch的训练
                        loader = self.data_curriculum(self.train_loader)  # curriculum part
                        num_steps = len(loader)#该epoch的训练集大小，课程学习的epoch不一定是全训练集！
                        if step >= num_steps:#如果当前batch的训练集大小已经超过了该epoch的训练集大小，就结束该epoch的训练，进入下一个epoch
                            break

                        data = next(iter(loader))
                        inputs = data[0].to(self.device)
                        labels = data[1].to(self.device)
                        indices = data[2].to(self.device)

                        self.optimizer.zero_grad()
                        outputs = net(inputs)
                        loss = self.loss_curriculum(  # curriculum part
                            self.criterion, outputs, labels, indices)
                        loss.backward()
                        self.optimizer.step()

                        train_loss += loss.item()
                        _, predicted = outputs.max(dim=1)
                        correct += predicted.eq(labels).sum().item()
                        total += labels.shape[0]

                        step += 1
                        steps_done_epoch = step

                        #每训练50个batch记录一次日志，或者每个epoch结束记录一次日志
                        if step % self.batch_log_interval == 0 or step == num_steps:
                            steps_done = step#用来算每个inv（50个batch)的平均损失
                            self.logger.info(
                                '[%3d]  Step %4d/%4d  Train Acc = %.4f  Loss = %.4f'
                                % (epoch + 1, steps_done, num_steps,
                                   correct / total, train_loss / steps_done))
                        
            #非adaptive算法：每个epoch的训练集大小不变，可以直接用for循环迭代训练集
            else:
                loader = self.data_curriculum(self.train_loader)  # curriculum part 
                num_steps = len(loader)
                for step, data in enumerate(loader):
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)
                    indices = data[2].to(self.device)

                    self.optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = self.loss_curriculum(  # curriculum part
                        self.criterion, outputs, labels, indices)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(dim=1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.shape[0]

                    steps_done_epoch = step + 1

                    if (step + 1) % self.batch_log_interval == 0 or (step + 1) == num_steps:
                    #因为for循环的step是从0开始的，所以要加1才能正确记录日志
                        steps_done = step + 1#用来算每个inv（50个batch)的平均损失
                        self.logger.info(
                            '[%3d]  Step %4d/%4d  Train Acc = %.4f  Loss = %.4f'
                            % (epoch + 1, steps_done, num_steps,
                               correct / total, train_loss / steps_done))

            #每个epoch结束后更新学习率（cosine下降），并评估训练集损失，训练时间
            self.lr_scheduler.step()
            self.logger.info(
                '[%3d]  Train data = %6d  Train Acc = %.4f  Loss = %.4f  Time = %.2f'
                % (epoch + 1, total, correct / total, train_loss / max(steps_done_epoch, 1), time.time() - t))

            #验证集评估模型性能（每个epoch结束后），并保存最佳模型
            if (epoch + 1) % self.log_interval == 0:
                valid_acc = self._valid(self.valid_loader)
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(net.state_dict(), os.path.join(self.log_dir, 'net.pkl'))
                self.logger.info(
                    '[%3d]  Valid data = %6d  Valid Acc = %.4f'
                    % (epoch + 1, len(self.valid_loader.dataset), valid_acc))
        total_time = time.time() - t0
        self.logger.info('Training Finished. Total Time = %.2f' % (total_time))

    def _valid(self, loader):
        total = 0
        correct = 0

        self.net.eval()
        with torch.no_grad():
            for data in loader:
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += predicted.eq(labels).sum().item()
        return correct / total
    
    #设置种子+训练模型
    def fit(self):
        set_random(self.random_seed)
        self._train()

    #评估：验证集，测试集
    def evaluate(self, net_dir=None):
        self._load_best_net(net_dir)
        valid_acc = self._valid(self.valid_loader)
        test_acc = self._valid(self.test_loader)
        self.logger.info('Best Valid Acc = %.4f and Final Test Acc = %.4f' % (valid_acc, test_acc))
        return test_acc

    def export(self, net_dir=None):
        self._load_best_net(net_dir)
        return self.net

    def _load_best_net(self, net_dir):
        if net_dir is None: net_dir = self.log_dir
        net_file = os.path.join(net_dir, 'net.pkl')
        assert os.path.exists(net_file), 'Assert Error: the net file does not exist'
        # Load only tensor weights to avoid unsafe pickle deserialization.
        state_dict = torch.load(net_file, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
