import math
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from .base import BaseTrainer, BaseCL



class Adaptive(BaseCL):
    """
    
    Adaptive Curriculum Learning. https://openaccess.thecvf.com/content/ICCV2021/papers/Kong_Adaptive_Curriculum_Learning_ICCV_2021_paper.pdf
    """
    def __init__(self, num_classes, pace_p, pace_q, pace_r, inv,
                 alpha, lambda1, lambda1_decay, bottom_lambda1, pretrained_net):
        super(Adaptive, self).__init__()

        self.name = 'adaptive'

        self.epoch = 0
        self.batch = 0
        self.pace_p = pace_p
        self.epoch_size = pace_p
        self.pace_q = pace_q
        self.pace_r = pace_r
        self.inv = inv
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda1_decay = lambda1_decay
        self.bottom_lambda1 = bottom_lambda1
        self.num_classes = num_classes
        self.pretrained_model = pretrained_net


    def data_prepare(self, loader):
        self.dataloader = loader
        self.dataset = self.CLDataset(loader.dataset)
        self.data_size = len(self.dataset)
        self.batch_size = loader.batch_size
        self.n_batches = (self.data_size - 1) // self.batch_size + 1
        # 课程是否已经扩展到全训练集。
        self.curriculum_finished = False


    def model_prepare(self, net, device, epochs, criterion, optimizer, lr_scheduler):
        self.device = device
        self.model = net
        self.critertion = criterion
        self.total_epoch = epochs
    
#按照课程抓取数据，并且在每个inv(=50个batch)结束后更新难度！
    def data_curriculum(self, loader):
        if self.epoch == 0 and self.batch == 0:
            self.pretrained_model.to(self.device)
            self.difficulty = torch.Tensor().to(self.device)
            self.pretrained_output = torch.Tensor().to(self.device)
            self.data_indice = torch.arange(self.data_size)
            self.crossEntrophy = torch.nn.CrossEntropyLoss(reduction='none')
            self.KLloss = torch.nn.KLDivLoss(reduction='batchmean')
            self._set_initial_difficulty()
            self.pretrained_difficulty = self.difficulty

        #训练集扩张公式
        self.epoch_size = self.data_size * min(
            self.pace_p * (self.pace_q ** int(math.floor(self.batch / self.pace_r))),
            1)
        self.epoch_size = int(self.epoch_size)
        
        #当课程已经扩展到全训练集(如 CIFAR-10 的 45000 样本)时，
        #跳过排序、Subset 构建和难度更新，直接返回完整训练集。
        if self.epoch_size == self.data_size:
            self.curriculum_finished = True
            dataloader = DataLoader(
                self.dataset,
                batch_size=loader.batch_size,
                shuffle=True,
                num_workers=loader.num_workers,
                pin_memory=loader.pin_memory,
            
            )

            self.batch += 1
            if self.batch % self.n_batches == 0:
                self.epoch += 1

            #保持lambda1的更新节奏，但不再执行全量difficulty测量
            if self.batch % self.inv == 0 and self.lambda1_decay is not None:
                self.lambda1 = max(self.bottom_lambda1, self.lambda1 - self.lambda1_decay)

            return dataloader

        #根据难度排序，选择前epoch_size个数据进行训练！
        data_sort = torch.argsort(self.difficulty)
        self.data_indice = data_sort[0 : self.epoch_size]
        dataset = Subset(self.dataset, self.data_indice)
        dataloader = DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=True,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            
        )

        self.batch += 1
        if self.batch % self.n_batches == 0:
            self.epoch += 1

        #更新难度,每隔一个inv(50个batch),并且要在500次迭代之后才更新难度！
        if self.batch % self.inv == 0 and (self.batch+1)>500:
            self._difficulty_measurer()

            # gradually reduce lambda1 which is the balancing parameter controling how much the knowledge learned from the pretrained model
            if self.lambda1_decay is not None:
                self.lambda1 = max(self.bottom_lambda1, self.lambda1 - self.lambda1_decay)

        return dataloader

    def update_after_curriculum_finished_step(self):
        """在课程结束后的常规for-loop中，保持batch/epoch与lambda1更新节奏。"""
        if not self.curriculum_finished:
            return

        self.batch += 1
        if self.batch % self.n_batches == 0:
            self.epoch += 1

        if self.batch % self.inv == 0 and self.lambda1_decay is not None:
            self.lambda1 = max(self.bottom_lambda1, self.lambda1 - self.lambda1_decay)

    def _difficulty_measurer(self):
    
        current_difficulty = torch.Tensor().to(self.device)

        for step, data in enumerate(self.dataloader):
            with torch.no_grad():
                outputs = self.model(data[0].to(self.device))
            loss = self.crossEntrophy(outputs, data[1].to(self.device)).detach()
            current_difficulty = torch.cat((current_difficulty, loss), 0)
        
        self.difficulty = (1 - self.alpha) * self.difficulty + self.alpha * current_difficulty
        #自适应更新难度！    
        
    def loss_curriculum(self, criterion, outputs, labels, indices):
        losses = torch.mean(criterion(outputs, labels))
        if indices is None:
            raise RuntimeError('Adaptive loss requires sample indices from CLDataset.')
        epoch_pretrained_output = self.pretrained_output[indices.long()]
        epoch_pretrained_output = epoch_pretrained_output.view(-1, self.num_classes)

        epoch_pretrained_output = F.softmax(epoch_pretrained_output, dim=1)

        output = F.softmax(outputs, dim=1)
        kl_divergence = self.KLloss(output, epoch_pretrained_output)
        #把预训练模型的输出当作‘伪标签’，计算KL散度，蒸馏模型！

        losses = losses + self.lambda1 * kl_divergence
        #目标函数：减少损失和增加与预训练模型输出的相似度（蒸馏）！！！
        return losses      


    def _set_initial_difficulty(self):
    
        self.pretrained_model.eval()
        for step, data in enumerate(self.dataloader):
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)
            with torch.no_grad():
                outputs = self.pretrained_model(inputs)
            self.pretrained_output = torch.cat((self.pretrained_output, outputs), 0)
            loss = self.crossEntrophy(outputs, labels)
            self.difficulty = torch.cat((self.difficulty, loss), 0)





class AdaptiveTrainer(BaseTrainer):
    def __init__(self, data_name, net_name, device_name, num_epochs, random_seed,
                 num_classes, pace_p, pace_q, pace_r, inv,
                 alpha, lambda1, lambda1_decay, bottom_lambda1, pretrained_net):
        
        cl = Adaptive(num_classes, pace_p, pace_q, pace_r, inv,
                 alpha, lambda1, lambda1_decay, bottom_lambda1, pretrained_net)

        super(AdaptiveTrainer, self).__init__(
            data_name, net_name, device_name, num_epochs, random_seed, cl)
