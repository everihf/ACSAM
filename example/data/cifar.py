import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from utility.cutout import Cutout


CIFAR100_SUPERCLASS_SUBSETS = {
    "cifar100_aquatic_mammals": [
        "beaver",
        "dolphin",
        "otter",
        "seal",
        "whale",
    ],
    "cifar100_small_mammals": [
        "hamster",
        "mouse",
        "rabbit",
        "shrew",
        "squirrel",
    ],
    "cifar100_household_electrical_devices": [
        "clock",
        "keyboard",
        "lamp",
        "telephone",
        "television",
    ],
}


class LabelMappedSubset(Dataset):
    """
    Filter a torchvision-style dataset by selected fine labels and remap labels to 0..K-1.

    Keeps `data`, `targets`, `classes`, and `class_to_idx` attributes so downstream code
    (e.g., fixed curriculum ordering) can reuse the same interfaces.
    """

    def __init__(self, base_dataset, selected_fine_indices, selected_class_names):
        self.base_dataset = base_dataset
        self.selected_fine_indices = list(selected_fine_indices)
        self.classes = list(selected_class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self._fine_to_remap = {fine_idx: remap_idx for remap_idx, fine_idx in enumerate(self.selected_fine_indices)}

        selected_set = set(self.selected_fine_indices)
        self.indices = [i for i, t in enumerate(base_dataset.targets) if int(t) in selected_set]
        self.targets = [self._fine_to_remap[int(base_dataset.targets[i])] for i in self.indices]

        if hasattr(base_dataset, "data"):
            self.data = base_dataset.data[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        base_index = self.indices[index]
        x, _ = self.base_dataset[base_index]
        y = self.targets[index]
        return x, y


class Cifar:
    def __init__(self, batch_size, num_workers, dataset="cifar10", use_data_augmentation=True):
        self.dataset = dataset.lower()
        self.use_data_augmentation = bool(use_data_augmentation)
        supported_datasets = {"cifar10", "cifar100"} | set(CIFAR100_SUPERCLASS_SUBSETS.keys())
        if self.dataset not in supported_datasets:
            raise ValueError(
                f"Unsupported dataset: {dataset}. "
                f"Use one of: {sorted(supported_datasets)}."
            )

        is_cifar100_subset = self.dataset in CIFAR100_SUPERCLASS_SUBSETS
        base_dataset_class = torchvision.datasets.CIFAR100 if (self.dataset == "cifar100" or is_cifar100_subset) else torchvision.datasets.CIFAR10

        if is_cifar100_subset:
            raw_train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
            class_to_idx = {name: idx for idx, name in enumerate(raw_train_set.classes)}
            selected_class_names = CIFAR100_SUPERCLASS_SUBSETS[self.dataset]
            selected_fine_indices = [class_to_idx[name] for name in selected_class_names]
            mean, std = self._get_subset_statistics(raw_train_set.data, raw_train_set.targets, selected_fine_indices)
        else:
            mean, std = self._get_statistics(base_dataset_class)

        train_transform_steps = []
        if self.use_data_augmentation:
            train_transform_steps.extend([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
            ])
        train_transform_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        if self.use_data_augmentation:
            train_transform_steps.append(Cutout())
        train_transform = transforms.Compose(train_transform_steps)

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_set = base_dataset_class(root="./data", train=True, download=True, transform=train_transform)
        test_set = base_dataset_class(root="./data", train=False, download=True, transform=test_transform)

        if is_cifar100_subset:
            train_set = LabelMappedSubset(
                base_dataset=train_set,
                selected_fine_indices=selected_fine_indices,
                selected_class_names=selected_class_names,
            )
            test_set = LabelMappedSubset(
                base_dataset=test_set,
                selected_fine_indices=selected_fine_indices,
                selected_class_names=selected_class_names,
            )

        self.train_dataset = train_set
        self.test_dataset = test_set

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classes = train_set.classes

    def _get_statistics(self, dataset_class):
        train_set = dataset_class(root="./data", train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

    def _get_subset_statistics(self, images_hwc_uint8, fine_targets, selected_fine_indices):
        selected_set = set(int(x) for x in selected_fine_indices)
        mask = torch.tensor([int(t) in selected_set for t in fine_targets], dtype=torch.bool)
        images = torch.from_numpy(images_hwc_uint8)[mask]  # [N, H, W, C], uint8
        images = images.float().permute(0, 3, 1, 2) / 255.0
        return images.mean(dim=[0, 2, 3]), images.std(dim=[0, 2, 3])
