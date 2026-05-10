from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


class IndexedDataset(Dataset):
    """Wrap a dataset to return sample indices alongside (x, y)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index

    def __len__(self):
        return len(self.dataset)


@dataclass
class FixedCurriculumConfig:
    batch_increase: int
    increase_amount: float
    starting_percent: float
    curriculum_type: str
    use_balance_order: bool = False


class FixedCurriculum:
    """
    Fixed curriculum learning with a static sample order and exponential pacing.

    It mirrors the fixed_curriculum_learning behavior:
    - Build one static sample order once (easy->hard).
    - Grow available subset size by exponential pacing.
    - Support curriculum / anti / random ordering variants.
    """

    def __init__(self, train_dataset, ordered_indices: Iterable[int], config: FixedCurriculumConfig):
        self.indexed_dataset = IndexedDataset(train_dataset)
        self.data_size = len(self.indexed_dataset)

        self.batch_increase = max(1, int(config.batch_increase))
        self.increase_amount = float(config.increase_amount)
        self.starting_percent = float(config.starting_percent)
        self.curriculum_type = str(config.curriculum_type).lower()
        self.use_balance_order = bool(config.use_balance_order)

        if self.curriculum_type not in {"curriculum", "anti", "random"}:
            raise ValueError(f"Unsupported curriculum_type: {self.curriculum_type}")
        if self.increase_amount < 1.0:
            raise ValueError("increase_amount must be >= 1.0 for fixed curriculum pacing.")
        if not (0.0 < self.starting_percent <= 1.0):
            raise ValueError("starting_percent must be within (0, 1].")

        ordered = torch.as_tensor(list(ordered_indices), dtype=torch.long)
        if ordered.numel() != self.data_size:
            raise ValueError(
                f"ordered_indices size mismatch: got {ordered.numel()}, expected {self.data_size}."
            )

        if self.curriculum_type == "anti":
            ordered = torch.flip(ordered, dims=[0])
        elif self.curriculum_type == "random":
            ordered = ordered[torch.randperm(self.data_size)]
        self.ordered_indices = ordered
        self.class_labels = None
        self.num_classes = 0
        labels = getattr(train_dataset, "targets", None)
        if labels is not None:
            labels_tensor = torch.as_tensor(list(labels), dtype=torch.long)
            if int(labels_tensor.numel()) == self.data_size:
                self.class_labels = labels_tensor
                self.num_classes = int(labels_tensor.max().item()) + 1 if labels_tensor.numel() > 0 else 0

        self.global_batch = 0
        self.curriculum_finished = False
        self._full_loader = None
        self.last_candidate_histogram = None

    def initialize(self, *_, **__):
        # Kept for API compatibility with AdaptiveCurriculum.
        return

    def _current_percent(self) -> float:
        if self.global_batch <= 0:
            return self.starting_percent
        growth_steps = self.global_batch // self.batch_increase
        return min(self.starting_percent * (self.increase_amount ** growth_steps), 1.0)

    def current_epoch_size(self) -> int:
        return max(1, int(torch.ceil(torch.tensor(self.data_size * self._current_percent())).item()))

    def build_dataloader(self, batch_size, num_workers, pin_memory):
        if self.curriculum_finished:
            return self.build_full_dataloader(batch_size, num_workers, pin_memory)

        epoch_size = self.current_epoch_size()
        if epoch_size >= self.data_size:
            self.curriculum_finished = True
            return self.build_full_dataloader(batch_size, num_workers, pin_memory)

        selected_indices = self.ordered_indices[:epoch_size].cpu().tolist()
        self.snapshot_candidate_histogram(torch.as_tensor(selected_indices, dtype=torch.long))
        subset = Subset(self.indexed_dataset, selected_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def build_full_dataloader(self, batch_size, num_workers, pin_memory):
        self.snapshot_candidate_histogram(self.ordered_indices)
        if self._full_loader is None:
            self._full_loader = DataLoader(
                self.indexed_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return self._full_loader

    def update_after_batch(self, *_, **__):
        self.global_batch += 1

    def _current_candidate_indices(self) -> torch.Tensor:
        if self.curriculum_finished:
            return self.ordered_indices

        epoch_size = self.current_epoch_size()
        if epoch_size >= self.data_size:
            self.curriculum_finished = True
            return self.ordered_indices
        return self.ordered_indices[:epoch_size]

    def snapshot_candidate_histogram(self, candidate_indices: torch.Tensor = None):
        if candidate_indices is None:
            candidate_indices = self._current_candidate_indices()

        candidate_indices = candidate_indices.long()
        candidate_count = int(candidate_indices.numel())
        class_counts = []
        if self.class_labels is not None:
            candidate_labels = self.class_labels[candidate_indices]
            counts_tensor = torch.bincount(candidate_labels, minlength=self.num_classes)[:self.num_classes]
            class_counts = [int(count.item()) for count in counts_tensor.cpu()]

        snapshot = {
            "global_batch": int(self.global_batch),
            "candidate_size": candidate_count,
            "curriculum_finished": bool(self.curriculum_finished),
            "class_counts": class_counts,
        }
        self.last_candidate_histogram = snapshot
        return snapshot

    def sample_batch(self, batch_size: int):
        """
        Sample one training batch using the current global-batch curriculum state.

        Sampling is random within the currently visible prefix of ordered indices.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        candidate_indices = self._current_candidate_indices()
        candidate_count = int(candidate_indices.numel())
        if candidate_count <= 0:
            raise RuntimeError("No candidate samples available for fixed curriculum batch sampling.")
        self.snapshot_candidate_histogram(candidate_indices)

        pick_count = min(batch_size, candidate_count)
        selected_positions = torch.randperm(candidate_count)[:pick_count]
        selected_indices = candidate_indices[selected_positions].tolist()

        samples = [self.indexed_dataset[int(sample_idx)] for sample_idx in selected_indices]
        inputs = torch.stack([sample[0] for sample in samples], dim=0)
        targets = torch.as_tensor([int(sample[1]) for sample in samples], dtype=torch.long)
        indices = torch.as_tensor([int(sample[2]) for sample in samples], dtype=torch.long)
        return inputs, targets, indices


def rank_samples_by_confidence(
    model: torch.nn.Module,
    train_dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> torch.Tensor:
    """
    Return sample indices sorted by predicted true-class probability (easy->hard).
    """
    indexed_dataset = IndexedDataset(train_dataset)
    loader = DataLoader(
        indexed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    scores = torch.zeros(len(indexed_dataset), device=device)
    model.eval()
    with torch.no_grad():
        for inputs, targets, indices in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            indices = indices.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            scores[indices] = true_class_probs

    ordered = torch.argsort(scores, descending=True)
    return ordered.cpu()


def balance_order_by_class(ordered_indices: torch.Tensor, labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Interleave easy-to-hard ordering across classes to reduce early class skew.
    """
    if ordered_indices.ndim != 1:
        raise ValueError("ordered_indices must be 1D.")

    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    if labels_tensor.numel() != ordered_indices.numel():
        raise ValueError("labels length must match ordered_indices length.")

    class_positions = []
    ordered_labels = labels_tensor[ordered_indices]
    for class_id in range(num_classes):
        positions = torch.nonzero(ordered_labels == class_id, as_tuple=False).squeeze(1)
        class_positions.append(positions)

    max_len = max((positions.numel() for positions in class_positions), default=0)
    interleaved_positions = []
    for row in range(max_len):
        row_positions = []
        for positions in class_positions:
            if row < positions.numel():
                row_positions.append(int(positions[row].item()))
        row_positions.sort()
        interleaved_positions.extend(row_positions)

    interleaved_positions_tensor = torch.as_tensor(interleaved_positions, dtype=torch.long)
    return ordered_indices[interleaved_positions_tensor]


class RawImageDataset(Dataset):
    """Dataset wrapper for raw HWC uint8 images and integer labels."""

    def __init__(self, images, targets, transform):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.images[index]
        target = int(self.targets[index])
        image = self.transform(image)
        return image, target, index


def _get_project_data_dir() -> Path:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _get_raw_images_and_targets(train_dataset):
    if not hasattr(train_dataset, "data") or not hasattr(train_dataset, "targets"):
        raise ValueError(
            "Inception+SVM ranking currently expects a torchvision-style CIFAR dataset "
            "with '.data' and '.targets' attributes."
        )
    return train_dataset.data, list(train_dataset.targets)


def _build_inception_feature_extractor(device):
    try:
        import torchvision.models as tv_models
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for inception_svm ordering. Install torchvision first."
        ) from exc

    # Keep downloaded Inception weights under project ./data
    torch.hub.set_dir(str(_get_project_data_dir()))

    try:
        weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
        # With pretrained weights, newer torchvision enforces aux_logits=True at build time.
        model = tv_models.inception_v3(weights=weights, transform_input=False)
    except AttributeError:
        model = tv_models.inception_v3(pretrained=True, transform_input=False)

    # We only need the main feature path for transfer values.
    if hasattr(model, "AuxLogits"):
        model.AuxLogits = None
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def _build_inception_preprocess():
    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for inception_svm ordering. Install torchvision first."
        ) from exc

    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _extract_inception_features(
    train_dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = _get_raw_images_and_targets(train_dataset)
    preprocess = _build_inception_preprocess()
    raw_dataset = RawImageDataset(images=images, targets=targets, transform=preprocess)
    loader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = _build_inception_feature_extractor(device)
    all_features = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            all_features.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())
    features = torch.cat(all_features, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    return features, targets_tensor


def rank_samples_by_inception_svm(
    train_dataset,
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    cache_dir=None,
    svm_kernel: str = "rbf",
    svm_c: float = 1.0,
    svm_gamma: str = "scale",
    use_cache: bool = True,
    svm_backend: str = "auto",
) -> torch.Tensor:
    """
    Rank train samples via Inception transfer features + SVM probability of true class.
    """
    import numpy as np

    scores, classes, targets_np = _compute_inception_svm_scores(
        train_dataset=train_dataset,
        dataset_name=dataset_name,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        cache_dir=cache_dir,
        svm_kernel=svm_kernel,
        svm_c=svm_c,
        svm_gamma=svm_gamma,
        use_cache=use_cache,
        svm_backend=svm_backend,
    )

    class_to_col = {int(cls): idx for idx, cls in enumerate(classes.tolist())}
    target_cols = np.asarray([class_to_col[int(t)] for t in targets_np], dtype=np.int64)
    hardness = scores[np.arange(scores.shape[0]), target_cols]
    ordered = np.argsort(-hardness)
    return torch.as_tensor(ordered, dtype=torch.long)


def build_inception_svm_teacher_logits(
    train_dataset,
    dataset_name: str,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    cache_dir=None,
    svm_kernel: str = "rbf",
    svm_c: float = 1.0,
    svm_gamma: str = "scale",
    use_cache: bool = True,
    svm_backend: str = "auto",
) -> torch.Tensor:
    """
    Build per-sample pseudo teacher logits from Inception features + SVM probabilities.

    Returned tensor shape: [N, num_classes], ordered by dataset sample index.
    """
    import numpy as np

    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")

    scores, classes, _ = _compute_inception_svm_scores(
        train_dataset=train_dataset,
        dataset_name=dataset_name,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        cache_dir=cache_dir,
        svm_kernel=svm_kernel,
        svm_c=svm_c,
        svm_gamma=svm_gamma,
        use_cache=use_cache,
        svm_backend=svm_backend,
    )

    probs = np.zeros((scores.shape[0], num_classes), dtype=np.float32)
    for col_idx, class_id in enumerate(classes.tolist()):
        class_id = int(class_id)
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"SVM predicted class id {class_id} is outside [0, {num_classes - 1}]."
            )
        probs[:, class_id] = scores[:, col_idx].astype(np.float32, copy=False)

    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, a_min=1e-12, a_max=None)
    probs = probs / row_sums

    logits = torch.log(torch.from_numpy(np.clip(probs, a_min=1e-12, a_max=1.0)))
    return logits


def _to_numpy_array(value, np):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _fit_svm_predict_proba(
    features_np,
    targets_np,
    svm_kernel: str,
    svm_c: float,
    svm_gamma: str,
    svm_backend: str,
):
    import numpy as np

    backend = str(svm_backend).lower()
    if backend not in {"auto", "cuml", "sklearn"}:
        raise ValueError(
            f"Unsupported svm_backend={svm_backend}. Expected one of: auto, cuml, sklearn."
        )

    backends = ["cuml", "sklearn"] if backend == "auto" else [backend]
    last_exc = None

    for current_backend in backends:
        if current_backend == "cuml":
            try:
                import cupy as cp
                from cuml.svm import SVC as cuSVC

                features_gpu = cp.asarray(features_np, dtype=cp.float32)
                targets_gpu = cp.asarray(targets_np, dtype=cp.int32)
                clf = cuSVC(
                    probability=True,
                    kernel=svm_kernel,
                    C=float(svm_c),
                    gamma=svm_gamma,
                )
                clf.fit(features_gpu, targets_gpu)
                scores = _to_numpy_array(clf.predict_proba(features_gpu), np).astype(np.float32, copy=False)
                classes = _to_numpy_array(clf.classes_, np).astype(np.int64, copy=False)
                return scores, classes, "cuml"
            except Exception as exc:  # pragma: no cover - depends on local CUDA/cuml runtime
                last_exc = exc
                if backend == "cuml":
                    raise RuntimeError(
                        "Failed to run GPU SVM with cuML. "
                        "Please ensure RAPIDS cuML/cupy are installed and CUDA-compatible."
                    ) from exc
                warnings.warn(
                    f"cuML backend unavailable ({exc}); falling back to sklearn SVM.",
                    RuntimeWarning,
                )
                continue

        if current_backend == "sklearn":
            try:
                from sklearn import svm as sk_svm
            except ImportError as exc:
                last_exc = exc
                raise RuntimeError(
                    "scikit-learn is required for sklearn SVM backend. Install scikit-learn first."
                ) from exc

            clf = sk_svm.SVC(probability=True, kernel=svm_kernel, C=svm_c, gamma=svm_gamma)
            clf.fit(features_np, targets_np)
            scores = clf.predict_proba(features_np).astype(np.float32, copy=False)
            classes = np.asarray(clf.classes_, dtype=np.int64)
            return scores, classes, "sklearn"

    raise RuntimeError("No available SVM backend.") from last_exc


def _compute_inception_svm_scores(
    train_dataset,
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    cache_dir=None,
    svm_kernel: str = "rbf",
    svm_c: float = 1.0,
    svm_gamma: str = "scale",
    use_cache: bool = True,
    svm_backend: str = "auto",
):
    """
    Compute (or load cached) per-sample class probabilities from Inception+SVM.
    Returns: scores[N, C], classes[C], targets[N].
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for inception_svm ordering.") from exc

    cache_root = Path(cache_dir) if cache_dir is not None else (_get_project_data_dir() / "inception_svm_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_tag = (
        f"{dataset_name}_b{str(svm_backend).lower()}_k{svm_kernel}"
        f"_c{str(svm_c).replace('.', 'p')}_g{str(svm_gamma)}"
    )
    feature_cache_path = cache_root / f"inception_features_{dataset_name}.pt"
    target_cache_path = cache_root / f"inception_targets_{dataset_name}.pt"
    score_cache_path = cache_root / f"inception_svm_scores_{cache_tag}.npz"

    features = None
    targets = None

    if use_cache and score_cache_path.exists():
        cached = np.load(score_cache_path, allow_pickle=False)
        scores = cached["scores"]
        svm_classes = cached["classes"]
        _, current_targets = _get_raw_images_and_targets(train_dataset)
        current_targets = np.asarray(current_targets, dtype=np.int64)
        if scores.shape[0] == current_targets.shape[0]:
            return scores, svm_classes, current_targets

    if use_cache and feature_cache_path.exists() and target_cache_path.exists():
        features = torch.load(feature_cache_path, map_location="cpu")
        targets = torch.load(target_cache_path, map_location="cpu")

    if features is None or targets is None:
        features, targets = _extract_inception_features(
            train_dataset=train_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if use_cache:
            torch.save(features, feature_cache_path)
            torch.save(targets, target_cache_path)

    features_np = features.numpy()
    targets_np = targets.numpy().astype("int64")

    scores, classes, _ = _fit_svm_predict_proba(
        features_np=features_np,
        targets_np=targets_np,
        svm_kernel=svm_kernel,
        svm_c=svm_c,
        svm_gamma=svm_gamma,
        svm_backend=svm_backend,
    )
    if use_cache:
        np.savez_compressed(score_cache_path, scores=scores, classes=classes)

    return scores, classes, targets_np
