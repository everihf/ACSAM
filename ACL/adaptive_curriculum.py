from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


class IndexedDataset(Dataset):
    """Wrap a dataset so every sample carries its original index."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError("Each dataset sample must be a tuple/list like (input, target).")
        inputs, target = sample[0], sample[1]
        return inputs, target, index

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass
class AdaptiveCurriculumConfig:
    """
    Paper-facing hyperparameters for Adaptive Curriculum Learning.

    The pacing rule follows the exponential schedule used in the original
    public ACL implementation:

        pool_size = n * min(pace_p * pace_q ** floor(step / pace_r), 1.0)

    where step is scaled by batch_size / pacing_reference_batch_size.
    """

    pace_p: float = 0.04
    pace_q: float = 1.1
    pace_r: int = 100
    inv: int = 50
    alpha: float = -0.01
    lambda_kl: float = 0.01
    lambda_kl_decay: Optional[float] = None
    lambda_kl_min: float = 0.0
    difficulty_warmup_batches: int = 500
    keep_class_balance: bool = True
    score_mode: str = "cross_entropy"
    pacing_reference_batch_size: float = 100.0

    def validate(self) -> None:
        if self.pace_p <= 0:
            raise ValueError(f"pace_p must be positive, got {self.pace_p}.")
        if self.pace_q < 1.0:
            raise ValueError(f"pace_q must be >= 1.0, got {self.pace_q}.")
        if self.pace_r <= 0:
            raise ValueError(f"pace_r must be positive, got {self.pace_r}.")
        if self.inv <= 0:
            raise ValueError(f"inv must be positive, got {self.inv}.")
        if self.score_mode not in {"cross_entropy", "confidence"}:
            raise ValueError(
                f"Unsupported score_mode={self.score_mode!r}. "
                "Use 'cross_entropy' or 'confidence'."
            )
        if self.pacing_reference_batch_size <= 0:
            raise ValueError(
                "pacing_reference_batch_size must be positive, "
                f"got {self.pacing_reference_batch_size}."
            )
        if self.lambda_kl_decay is not None and self.lambda_kl_decay < 0:
            raise ValueError("lambda_kl_decay must be non-negative when provided.")
        if self.lambda_kl_min < 0:
            raise ValueError("lambda_kl_min must be non-negative.")
        if self.lambda_kl_min > self.lambda_kl:
            raise ValueError("lambda_kl_min must be <= lambda_kl.")


class AdaptiveCurriculumLearning:
    """
    Reproduction of Adaptive Curriculum Learning (ICCV 2021).

    Core paper mechanics:
    - initialize a pseudo-ideal score s0 from a pretrained source;
    - update the score every `inv` mini-batches via
      s <- (1 - alpha) * s + alpha * s_cur;
    - sample each training batch uniformly from the easiest p(m) examples;
    - optimize CE + lambda * KL(student || teacher).

    The class is model-agnostic as long as the dataset returns (input, label)
    and the model returns class logits.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        num_classes: int,
        device: torch.device | str,
        config: AdaptiveCurriculumConfig,
        teacher_model: Optional[torch.nn.Module] = None,
        teacher_logits_by_index: Optional[torch.Tensor] = None,
        initial_difficulty: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.config.validate()

        self.device = torch.device(device)
        self.num_classes = int(num_classes)
        self.indexed_dataset = IndexedDataset(train_dataset)
        self.data_size = len(self.indexed_dataset)

        self.teacher_model = None
        if teacher_model is not None:
            self.teacher_model = deepcopy(teacher_model).to(self.device)
            self.teacher_model.eval()

        self.teacher_logits = None
        if teacher_logits_by_index is not None:
            expected_shape = (self.data_size, self.num_classes)
            if tuple(teacher_logits_by_index.shape) != expected_shape:
                raise ValueError(
                    "teacher_logits_by_index shape mismatch: "
                    f"got {tuple(teacher_logits_by_index.shape)}, expected {expected_shape}."
                )
            self.teacher_logits = teacher_logits_by_index.detach().to(self.device)

        self.initial_difficulty = None
        if initial_difficulty is not None:
            if tuple(initial_difficulty.shape) != (self.data_size,):
                raise ValueError(
                    "initial_difficulty shape mismatch: "
                    f"got {tuple(initial_difficulty.shape)}, expected {(self.data_size,)}."
                )
            self.initial_difficulty = initial_difficulty.detach().to(self.device).float()

        if self.config.lambda_kl > 0 and self.teacher_model is None and self.teacher_logits is None:
            raise ValueError(
                "lambda_kl > 0 requires teacher_model or teacher_logits_by_index "
                "for the KL term."
            )

        self.class_labels = self._extract_labels(train_dataset)
        self.global_batch = 0
        self.curriculum_finished = False
        self.batch_size = None
        self.num_workers = 0
        self.pin_memory = False

        self.difficulty = torch.zeros(self.data_size, device=self.device)
        self._initialized = False

    def initialize(
        self,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if self._initialized:
            return

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        if self.teacher_logits is None and self.teacher_model is not None:
            self.teacher_logits = self._collect_logits(self.teacher_model)
            self.teacher_model = None

        if self.initial_difficulty is not None:
            self.difficulty.copy_(self.initial_difficulty)
        elif self.teacher_logits is not None:
            targets = self.class_labels
            if targets is None:
                raise RuntimeError(
                    "Could not infer labels from the dataset, so initial difficulty "
                    "cannot be computed from teacher logits automatically."
                )
            self.difficulty.copy_(self._score_from_logits(self.teacher_logits, targets))
        else:
            raise ValueError(
                "initialize() needs either initial_difficulty, teacher_model, "
                "or teacher_logits_by_index."
            )

        self._initialized = True
        self.curriculum_finished = self.current_pool_size() >= self.data_size

    def current_pool_size(self) -> int:
        self._require_initialized()
        scaled_step = self.global_batch * (
            self.batch_size / self.config.pacing_reference_batch_size
        )
        growth_steps = int(math.floor(scaled_step / self.config.pace_r))
        pool_ratio = min(self.config.pace_p * (self.config.pace_q ** growth_steps), 1.0)
        return max(1, int(math.ceil(self.data_size * pool_ratio)))

    def sample_pool_indices(self) -> torch.Tensor:
        self._require_initialized()

        ordered_indices = torch.argsort(self.difficulty)
        pool_size = self.current_pool_size()
        self.curriculum_finished = pool_size >= self.data_size

        if pool_size >= self.data_size:
            return ordered_indices
        if not self.config.keep_class_balance or self.class_labels is None:
            return ordered_indices[:pool_size]
        return self._balanced_prefix(ordered_indices, pool_size)

    def build_dataloader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ) -> DataLoader:
        self._require_initialized()

        effective_batch_size = int(batch_size or self.batch_size)
        effective_num_workers = self.num_workers if num_workers is None else int(num_workers)
        effective_pin_memory = self.pin_memory if pin_memory is None else bool(pin_memory)

        selected_indices = self.sample_pool_indices()
        if int(selected_indices.numel()) >= self.data_size:
            self.curriculum_finished = True
            return self.build_full_dataloader(
                batch_size=effective_batch_size,
                num_workers=effective_num_workers,
                pin_memory=effective_pin_memory,
            )

        subset = Subset(self.indexed_dataset, selected_indices.cpu())
        return DataLoader(
            subset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=effective_num_workers,
            pin_memory=effective_pin_memory,
        )

    def build_full_dataloader(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
    ) -> DataLoader:
        self._require_initialized()

        effective_batch_size = int(batch_size or self.batch_size)
        effective_num_workers = self.num_workers if num_workers is None else int(num_workers)
        effective_pin_memory = self.pin_memory if pin_memory is None else bool(pin_memory)

        return DataLoader(
            self.indexed_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=effective_num_workers,
            pin_memory=effective_pin_memory,
        )

    def sample_batch(self, batch_size: Optional[int] = None):
        self._require_initialized()

        effective_batch_size = int(batch_size or self.batch_size)
        if effective_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {effective_batch_size}.")

        pool_indices = self.sample_pool_indices()
        pool_count = int(pool_indices.numel())
        if pool_count <= 0:
            raise RuntimeError("Curriculum sample pool is empty.")

        if pool_count >= effective_batch_size:
            chosen_positions = torch.randperm(pool_count, device=pool_indices.device)[:effective_batch_size]
        else:
            chosen_positions = torch.randint(
                low=0,
                high=pool_count,
                size=(effective_batch_size,),
                device=pool_indices.device,
            )

        chosen_indices = pool_indices[chosen_positions].cpu().tolist()
        samples = [self.indexed_dataset[int(index)] for index in chosen_indices]
        inputs = torch.stack([sample[0] for sample in samples], dim=0)
        targets = torch.as_tensor([int(sample[1]) for sample in samples], dtype=torch.long)
        indices = torch.as_tensor([int(sample[2]) for sample in samples], dtype=torch.long)
        return inputs, targets, indices

    def curriculum_loss(
        self,
        per_sample_losses: torch.Tensor,
        student_logits: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        self._require_initialized()

        base_loss = per_sample_losses.mean()
        if self.teacher_logits is None or self.config.lambda_kl <= 0:
            return base_loss

        indices = indices.long().to(self.device)
        teacher_probs = F.softmax(self.teacher_logits[indices], dim=1)
        student_log_probs = F.log_softmax(student_logits, dim=1)
        kl_term = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return base_loss + self.config.lambda_kl * kl_term

    def update_after_batch(self, student_model: torch.nn.Module) -> None:
        self._require_initialized()

        self.global_batch += 1
        self.curriculum_finished = self.current_pool_size() >= self.data_size

        if self.global_batch % self.config.inv == 0:
            if self.global_batch > self.config.difficulty_warmup_batches:
                current_scores = self._measure_student_difficulty(student_model)
                self.difficulty = (
                    (1.0 - self.config.alpha) * self.difficulty
                    + self.config.alpha * current_scores
                )

            if self.config.lambda_kl_decay is not None:
                self.config.lambda_kl = max(
                    self.config.lambda_kl_min,
                    self.config.lambda_kl - self.config.lambda_kl_decay,
                )

    def _collect_logits(self, model: torch.nn.Module) -> torch.Tensor:
        loader = DataLoader(
            self.indexed_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        logits_by_index = torch.zeros(
            self.data_size,
            self.num_classes,
            device=self.device,
        )

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for inputs, _, indices in loader:
                inputs = inputs.to(self.device)
                indices = indices.to(self.device)
                logits_by_index[indices] = model(inputs)
        if was_training:
            model.train()
        return logits_by_index

    def _measure_student_difficulty(self, student_model: torch.nn.Module) -> torch.Tensor:
        loader = DataLoader(
            self.indexed_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        current_scores = torch.zeros(self.data_size, device=self.device)

        was_training = student_model.training
        student_model.eval()
        with torch.no_grad():
            for inputs, targets, indices in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                indices = indices.to(self.device)
                logits = student_model(inputs)
                current_scores[indices] = self._score_from_logits(logits, targets)
        if was_training:
            student_model.train()
        return current_scores

    def _score_from_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.score_mode == "cross_entropy":
            return F.cross_entropy(logits, targets, reduction="none")

        probabilities = F.softmax(logits, dim=1)
        true_class_probability = probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
        return 1.0 - true_class_probability

    def _balanced_prefix(self, ordered_indices: torch.Tensor, pool_size: int) -> torch.Tensor:
        class_counts = torch.bincount(self.class_labels, minlength=self.num_classes).float()
        desired = class_counts * (float(pool_size) / float(self.data_size))
        quotas = torch.floor(desired).long()

        remaining = int(pool_size - int(quotas.sum().item()))
        if remaining > 0:
            fractional = desired - quotas.float()
            ranking = torch.argsort(fractional, descending=True)
            for class_id in ranking.tolist():
                if remaining == 0:
                    break
                if quotas[class_id] < int(class_counts[class_id].item()):
                    quotas[class_id] += 1
                    remaining -= 1

        ordered_labels = self.class_labels[ordered_indices]
        selected = []
        for class_id in range(self.num_classes):
            take = int(quotas[class_id].item())
            if take <= 0:
                continue
            class_indices = ordered_indices[ordered_labels == class_id][:take]
            if class_indices.numel() > 0:
                selected.append(class_indices)

        if not selected:
            return ordered_indices[:pool_size]

        selected_indices = torch.cat(selected, dim=0)
        rank = torch.empty(self.data_size, dtype=torch.long, device=ordered_indices.device)
        rank[ordered_indices] = torch.arange(
            self.data_size,
            dtype=torch.long,
            device=ordered_indices.device,
        )
        return selected_indices[torch.argsort(rank[selected_indices])]

    def _extract_labels(self, dataset: Dataset) -> Optional[torch.Tensor]:
        labels = None
        if hasattr(dataset, "targets"):
            labels = getattr(dataset, "targets")
        elif hasattr(dataset, "labels"):
            labels = getattr(dataset, "labels")
        elif isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 2:
            labels = dataset.tensors[1]

        if labels is None:
            return None

        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        if labels_tensor.numel() != self.data_size:
            return None
        return labels_tensor.to(self.device)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call initialize(...) before using the curriculum.")
