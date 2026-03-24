import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class IndexedDataset(Dataset):
    """Attach sample indices so curriculum logic can map logits by sample id."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index

    def __len__(self):
        return len(self.dataset)


class AdaptiveCurriculum:
    """
    Adaptive curriculum scheduler + teacher distillation term.

    This class keeps model/data interfaces identical to the original SAM example:
    - Same model forward usage.
    - Same CIFAR dataset (only wraps train dataset with sample indices).
    - Optional teacher checkpoint from the same architecture.
    """

    def __init__(
        self,
        train_dataset,
        teacher_model,
        device,
        num_classes,
        pace_p,
        pace_q,
        pace_r,
        inv,
        alpha,
        lambda1,
        lambda1_decay,
        bottom_lambda1,
    ):
        self.device = device
        self.num_classes = num_classes

        self.indexed_dataset = IndexedDataset(train_dataset)
        self.data_size = len(self.indexed_dataset)

        self.pace_p = pace_p
        self.pace_q = pace_q
        self.pace_r = pace_r
        self.inv = inv

        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda1_decay = lambda1_decay
        self.bottom_lambda1 = bottom_lambda1
        # 防止 bottom_lambda1 > lambda1 时出现“衰减反而升高权重”的情况
        # 若发生，自动把下界收缩到当前 lambda1。
        if self.bottom_lambda1 > self.lambda1:
            self.bottom_lambda1 = self.lambda1

        self.teacher_model = deepcopy(teacher_model).to(device)
        self.teacher_model.eval()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

        self.global_batch = 0
        self.pretrained_output = torch.zeros(self.data_size, num_classes, device=device)
        self.difficulty = torch.zeros(self.data_size, device=device)
        self._initialized = False

    def initialize(self, batch_size, num_workers, pin_memory):
        if self._initialized:
            return

        loader = DataLoader(
            self.indexed_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.batch_size=batch_size

        with torch.no_grad():
            for inputs, targets, indices in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                indices = indices.to(self.device)

                logits = self.teacher_model(inputs)
                # 保存教师模型在每个样本上的输出，后续蒸馏直接按 index 对齐读取。
                self.pretrained_output[indices] = logits
                # 以教师模型的逐样本交叉熵作为初始难度（越小越“容易”）。
                self.difficulty[indices] = self.cross_entropy(logits, targets)

        self._initialized = True

    def current_epoch_size(self):
        # 训练集扩张公式：
        # epoch_size = N * min(pace_p * pace_q ^ floor(batch / pace_r), 1)
        growth = self.pace_p * (self.pace_q ** int(math.floor(self.global_batch * (self.batch_size / 100) / self.pace_r)))
        return int(self.data_size * min(growth, 1.0))

    def build_dataloader(self, batch_size, num_workers, pin_memory):
        epoch_size = max(1, self.current_epoch_size())
        if epoch_size >= self.data_size:
            # 当课程扩张到全训练集后，退化为普通全量训练。
            dataset = self.indexed_dataset
        else:
            # 根据难度排序，优先选择“容易样本”（最小 loss）进入当前课程。
            sorted_indices = torch.argsort(self.difficulty)
            dataset = Subset(self.indexed_dataset, sorted_indices[:epoch_size].cpu())

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def curriculum_loss(self, per_sample_losses, outputs, indices):
        # 基础监督损失（与原 SAM 训练一致：逐样本损失再求均值）。
        base_loss = per_sample_losses.mean()

        teacher_logits = self.pretrained_output[indices.long()]
        teacher_probs = F.softmax(teacher_logits, dim=1)
        student_log_probs = F.log_softmax(outputs, dim=1)
        kl_div = self.kl_loss(student_log_probs, teacher_probs)
        # 目标函数 = 监督损失 + lambda1 * 蒸馏 KL。
        # 其中 lambda1 控制从 teacher 迁移知识的强度。

        return base_loss + self.lambda1 * kl_div

    def update_after_batch(self, model, batch_size, num_workers, pin_memory):
        self.global_batch += 1

        # 更新难度：每隔 inv 个 batch，且达到 warmup（>500）后执行。
        should_update_difficulty = self.global_batch % self.inv == 0 and (self.global_batch + 1) > 500
        if should_update_difficulty:
            self._remeasure_difficulty(model, batch_size, num_workers, pin_memory)

        # 与原 ACL 逻辑一致：lambda1 可随训练逐步衰减到下界。
        if self.global_batch % self.inv == 0 and self.lambda1_decay is not None:
            self.lambda1 = max(self.bottom_lambda1, self.lambda1 - self.lambda1_decay)

    def _remeasure_difficulty(self, model, batch_size, num_workers, pin_memory):
        loader = DataLoader(
            self.indexed_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        model.eval()
        current_difficulty = torch.zeros(self.data_size, device=self.device)

        with torch.no_grad():
            for inputs, targets, indices in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                indices = indices.to(self.device)

                outputs = model(inputs)
                current_difficulty[indices] = self.cross_entropy(outputs, targets)

        model.train()
        # 自适应更新难度：difficulty <- (1-alpha)*old + alpha*current
        self.difficulty = (1 - self.alpha) * self.difficulty + self.alpha * current_difficulty
