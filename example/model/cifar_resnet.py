from typing import List, Optional, Type

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int,
        base_width: int = 16,
        stage_widths: Optional[List[int]] = None,
    ):
        super().__init__()
        self.in_planes = base_width
        widths = stage_widths or [base_width, base_width * 2, base_width * 4, base_width * 8]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        self.layer4 = None
        if len(layers) == 4:
            self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2)
            final_width = widths[3]
        else:
            final_width = widths[2]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_width * block.expansion, num_classes)

        self._initialize()

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [block(self.in_planes, planes, stride=stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def cifar_resnet18(num_classes: int) -> CifarResNet:
    return CifarResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=64,
    )


def cifar_resnet32(num_classes: int) -> CifarResNet:
    return CifarResNet(
        block=BasicBlock,
        layers=[5, 5, 5],
        num_classes=num_classes,
        base_width=16,
        stage_widths=[16, 32, 64],
    )
