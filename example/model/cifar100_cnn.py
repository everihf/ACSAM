import torch.nn as nn


class Cifar100CNN(nn.Module):
    """PyTorch counterpart of fixed_curriculum_learning/models/cifar100_model.py."""

    def __init__(
        self,
        num_classes: int,
        activation: str = "elu",
        dropout_1_rate: float = 0.25,
        dropout_2_rate: float = 0.5,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32, 3, activation, batch_norm),
            self._conv_block(32, 32, 3, activation, batch_norm),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_1_rate),
            self._conv_block(32, 64, 3, activation, batch_norm),
            self._conv_block(64, 64, 3, activation, batch_norm),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_1_rate),
            self._conv_block(64, 128, 3, activation, batch_norm),
            self._conv_block(128, 128, 3, activation, batch_norm),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_1_rate),
            self._conv_block(128, 256, 2, activation, batch_norm),
            self._conv_block(256, 256, 2, activation, batch_norm),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout_1_rate),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            self._activation(activation),
            nn.Dropout(p=dropout_2_rate),
            nn.Linear(512, num_classes),
        )

        self._initialize()

    @staticmethod
    def _activation(name: str) -> nn.Module:
        normalized = name.lower()
        if normalized == "elu":
            return nn.ELU(inplace=True)
        if normalized == "relu":
            return nn.ReLU(inplace=True)
        if normalized == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation: {name}")

    def _conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str,
        batch_norm: bool,
    ) -> nn.Sequential:
        layers = []
        padding = 1 if kernel_size == 3 else 0
        if kernel_size == 2:
            # Keras "same" padding for 2x2 kernels with stride=1.
            layers.append(nn.ZeroPad2d((0, 1, 0, 1)))
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=not batch_norm,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(self._activation(activation))
        return nn.Sequential(*layers)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
