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

    Teacher source can be either:
    - a neural teacher model (`teacher_model`), or
    - precomputed per-sample logits (`teacher_logits_by_index`), e.g. Inception+SVM.
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
        curriculum_type="curriculum",
        use_difficulty_sorting=True,
        use_balance_order=True,
        teacher_logits_by_index=None,
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
        # Backward compatibility for older callers that only pass use_difficulty_sorting.
        self.use_difficulty_sorting = bool(use_difficulty_sorting)
        if curriculum_type is None:
            self.curriculum_type = "curriculum" if self.use_difficulty_sorting else "random"
        else:
            self.curriculum_type = str(curriculum_type).lower()
        if self.curriculum_type not in {"curriculum", "anti", "random"}:
            raise ValueError(f"Unsupported curriculum_type: {self.curriculum_type}")
        self.use_balance_order = bool(use_balance_order)
        self.lambda1 = lambda1
        self.lambda1_decay = lambda1_decay
        self.bottom_lambda1 = bottom_lambda1
        if self.bottom_lambda1 > self.lambda1:
            self.bottom_lambda1 = self.lambda1

        labels = getattr(train_dataset, "targets", None)
        self.class_labels = None
        if labels is not None:
            labels_tensor = torch.as_tensor(list(labels), dtype=torch.long)
            if int(labels_tensor.numel()) == self.data_size:
                self.class_labels = labels_tensor.to(device)
            else:
                self.use_balance_order = False
        else:
            self.use_balance_order = False

        self.teacher_logits_by_index = None
        if teacher_logits_by_index is not None:
            expected_shape = (self.data_size, self.num_classes)
            if tuple(teacher_logits_by_index.shape) != expected_shape:
                raise ValueError(
                    "teacher_logits_by_index shape mismatch: "
                    f"got {tuple(teacher_logits_by_index.shape)}, expected {expected_shape}."
                )
            self.teacher_logits_by_index = teacher_logits_by_index.detach().to(device)

        self.teacher_model = None
        if teacher_model is not None:
            self.teacher_model = deepcopy(teacher_model).to(device)
            self.teacher_model.eval()

        if self.teacher_model is None and self.teacher_logits_by_index is None:
            raise ValueError("Either teacher_model or teacher_logits_by_index must be provided.")

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

        self.global_batch = 0
        self.pretrained_output = torch.zeros(self.data_size, num_classes, device=device)
        self.difficulty = torch.zeros(self.data_size, device=device)
        self.curriculum_finished = False
        self._full_loader = None
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
        self.batch_size = batch_size

        with torch.no_grad():
            if self.teacher_logits_by_index is not None:
                self.pretrained_output.copy_(self.teacher_logits_by_index)
                for _, targets, indices in loader:
                    targets = targets.to(self.device)
                    indices = indices.to(self.device)
                    logits = self.pretrained_output[indices]
                    self.difficulty[indices] = self.cross_entropy(logits, targets)
            else:
                for inputs, targets, indices in loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    indices = indices.to(self.device)

                    logits = self.teacher_model(inputs)
                    self.pretrained_output[indices] = logits
                    self.difficulty[indices] = self.cross_entropy(logits, targets)

        self._initialized = True

    def current_epoch_size(self):
        # Match fixed-curriculum expansion: expand by global batch milestones.
        if self.global_batch <= 0:
            current_percent = self.pace_p
        else:
            growth_steps = self.global_batch // max(1, int(self.pace_r))
            current_percent = self.pace_p * (self.pace_q ** growth_steps)
        current_percent = min(current_percent, 1.0)
        return max(1, int(math.ceil(self.data_size * current_percent)))

    def build_dataloader(self, batch_size, num_workers, pin_memory):
        if self.curriculum_finished:
            return self.build_full_dataloader(batch_size, num_workers, pin_memory)

        selected_indices = self._current_candidate_indices()
        if self.curriculum_finished or int(selected_indices.numel()) >= self.data_size:
            self.curriculum_finished = True
            return self.build_full_dataloader(batch_size, num_workers, pin_memory)

        dataset = Subset(self.indexed_dataset, selected_indices.cpu())
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def build_full_dataloader(self, batch_size, num_workers, pin_memory):
        if self._full_loader is None:
            self._full_loader = DataLoader(
                self.indexed_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return self._full_loader

    def _current_candidate_indices(self) -> torch.Tensor:
        if self.curriculum_finished:
            return torch.arange(self.data_size, device=self.device)

        epoch_size = max(1, self.current_epoch_size())
        if epoch_size >= self.data_size:
            self.curriculum_finished = True
            return torch.arange(self.data_size, device=self.device)

        if self.curriculum_type == "random":
            return torch.randperm(self.data_size, device=self.device)[:epoch_size]

        ordered_indices = torch.argsort(self.difficulty)
        if self.use_balance_order:
            ordered_indices = self._balance_order_by_class(ordered_indices)
        if self.curriculum_type == "anti":
            ordered_indices = torch.flip(ordered_indices, dims=[0])
        return ordered_indices[:epoch_size]

    def _balance_order_by_class(self, ordered_indices: torch.Tensor) -> torch.Tensor:
        if self.class_labels is None:
            return ordered_indices
        if ordered_indices.ndim != 1:
            return ordered_indices

        ordered_labels = self.class_labels[ordered_indices]
        class_positions = []
        for class_id in range(self.num_classes):
            positions = torch.nonzero(ordered_labels == class_id, as_tuple=False).squeeze(1)
            class_positions.append(positions)

        max_len = max((int(positions.numel()) for positions in class_positions), default=0)
        if max_len == 0:
            return ordered_indices

        interleaved_positions = []
        for row in range(max_len):
            row_positions = []
            for positions in class_positions:
                if row < positions.numel():
                    row_positions.append(int(positions[row].item()))
            row_positions.sort()
            interleaved_positions.extend(row_positions)

        if not interleaved_positions:
            return ordered_indices
        interleaved_positions_tensor = torch.as_tensor(
            interleaved_positions,
            dtype=torch.long,
            device=ordered_indices.device,
        )
        return ordered_indices[interleaved_positions_tensor]

    def sample_batch(self, batch_size: int):
        """
        Sample one training batch using the current global-batch curriculum state.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        candidate_indices = self._current_candidate_indices()
        candidate_count = int(candidate_indices.numel())
        if candidate_count <= 0:
            raise RuntimeError("No candidate samples available for adaptive curriculum batch sampling.")

        pick_count = min(batch_size, candidate_count)
        selected_positions = torch.randperm(candidate_count, device=candidate_indices.device)[:pick_count]
        selected_indices = candidate_indices[selected_positions].cpu().tolist()

        samples = [self.indexed_dataset[int(sample_idx)] for sample_idx in selected_indices]
        inputs = torch.stack([sample[0] for sample in samples], dim=0)
        targets = torch.as_tensor([int(sample[1]) for sample in samples], dtype=torch.long)
        indices = torch.as_tensor([int(sample[2]) for sample in samples], dtype=torch.long)
        return inputs, targets, indices

    def curriculum_loss(self, per_sample_losses, outputs, indices):
        base_loss = per_sample_losses.mean()

        teacher_logits = self.pretrained_output[indices.long()]
        teacher_probs = F.softmax(teacher_logits, dim=1)
        student_log_probs = F.log_softmax(outputs, dim=1)
        kl_div = self.kl_loss(student_log_probs, teacher_probs)

        return base_loss + self.lambda1 * kl_div

    def update_after_batch(self, model, batch_size, num_workers, pin_memory):
        self.global_batch += 1

        if self.curriculum_finished:
            if self.global_batch % self.inv == 0 and self.lambda1_decay is not None:
                self.lambda1 = max(self.bottom_lambda1, self.lambda1 - self.lambda1_decay)
            return

        should_update_difficulty = self.global_batch % self.inv == 0 and (self.global_batch + 1) > 150
        if should_update_difficulty:
            self._remeasure_difficulty(model, batch_size, num_workers, pin_memory)

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
        self.difficulty = (1 - self.alpha) * self.difficulty + self.alpha * current_difficulty
