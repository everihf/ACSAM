import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utility.cutout import Cutout


class Cifar:
    def __init__(self, batch_size, num_workers, dataset="cifar10"):
        self.dataset = dataset.lower()
        if self.dataset not in {"cifar10", "cifar100"}:
            raise ValueError(f"Unsupported dataset: {dataset}. Use 'cifar10' or 'cifar100'.")

        dataset_class = torchvision.datasets.CIFAR10 if self.dataset == "cifar10" else torchvision.datasets.CIFAR100
        mean, std = self._get_statistics(dataset_class)
        #自动计算数据集的均值（mean）和标准差（std），用于后面的 Normalize 标准化

        #数据增强||经典 CIFAR-10 baseline augmentation
        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = dataset_class(root="./data", train=True, download=True, transform=train_transform)
        test_set = dataset_class(root="./data", train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.classes = train_set.classes
        #CIFAR-10 的 classes 是：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    def _get_statistics(self, dataset_class):
        train_set = dataset_class(root="./data", train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])
    #把所有训练图像拼起来，计算 每个通道的 mean / std
    #自动计算数据集的均值（mean）和标准差（std），用于后面的 Normalize 标准化