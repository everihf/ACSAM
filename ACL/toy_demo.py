from __future__ import annotations

import argparse
import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    from adaptive_curriculum import AdaptiveCurriculumConfig, AdaptiveCurriculumLearning
else:
    from .adaptive_curriculum import AdaptiveCurriculumConfig, AdaptiveCurriculumLearning


class TinyMLP(nn.Module):
    def __init__(self, in_features: int = 2, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_toy_split(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def sample_class(center_x: float, center_y: float, count: int) -> torch.Tensor:
        points = torch.randn(count, 2, generator=generator) * 0.75
        points[:, 0] += center_x
        points[:, 1] += center_y
        return points

    train_per_cluster = 240
    val_per_cluster = 80

    train_c0 = torch.cat(
        [sample_class(-2.0, -2.0, train_per_cluster), sample_class(-1.0, 2.0, train_per_cluster)],
        dim=0,
    )
    train_c1 = torch.cat(
        [sample_class(2.0, 2.0, train_per_cluster), sample_class(1.0, -2.0, train_per_cluster)],
        dim=0,
    )

    val_c0 = torch.cat(
        [sample_class(-2.0, -2.0, val_per_cluster), sample_class(-1.0, 2.0, val_per_cluster)],
        dim=0,
    )
    val_c1 = torch.cat(
        [sample_class(2.0, 2.0, val_per_cluster), sample_class(1.0, -2.0, val_per_cluster)],
        dim=0,
    )

    train_x = torch.cat([train_c0, train_c1], dim=0)
    train_y = torch.cat(
        [torch.zeros(train_c0.size(0), dtype=torch.long), torch.ones(train_c1.size(0), dtype=torch.long)],
        dim=0,
    )
    val_x = torch.cat([val_c0, val_c1], dim=0)
    val_y = torch.cat(
        [torch.zeros(val_c0.size(0), dtype=torch.long), torch.ones(val_c1.size(0), dtype=torch.long)],
        dim=0,
    )

    train_perm = torch.randperm(train_x.size(0), generator=generator)
    val_perm = torch.randperm(val_x.size(0), generator=generator)

    train_dataset = TensorDataset(train_x[train_perm], train_y[train_perm])
    val_dataset = TensorDataset(val_x[val_perm], val_y[val_perm])
    return train_dataset, val_dataset


def train_supervised(
    model: nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataset: TensorDataset, device: torch.device, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
    return correct / max(1, total)


def run_demo(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    train_dataset, val_dataset = make_toy_split(seed=args.seed)

    teacher = TinyMLP(hidden=args.hidden).to(device)
    train_supervised(
        model=teacher,
        dataset=train_dataset,
        device=device,
        epochs=args.teacher_epochs,
        batch_size=args.batch_size,
        lr=args.teacher_lr,
    )

    config = AdaptiveCurriculumConfig(
        pace_p=args.pace_p,
        pace_q=args.pace_q,
        pace_r=args.pace_r,
        inv=args.inv,
        alpha=args.alpha,
        lambda_kl=args.lambda_kl,
        lambda_kl_decay=args.lambda_kl_decay,
        lambda_kl_min=args.lambda_kl_min,
        difficulty_warmup_batches=args.difficulty_warmup_batches,
        keep_class_balance=not args.disable_class_balance,
        score_mode=args.score_mode,
    )

    student = TinyMLP(hidden=args.hidden).to(device)
    curriculum = AdaptiveCurriculumLearning(
        train_dataset=train_dataset,
        num_classes=2,
        device=device,
        config=config,
        teacher_model=teacher,
    )
    curriculum.initialize(batch_size=args.batch_size)

    optimizer = torch.optim.SGD(student.parameters(), lr=args.student_lr, momentum=0.9)
    num_batches = int(math.ceil(len(train_dataset) / args.batch_size))

    print("ACL config:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    print()

    for epoch in range(1, args.student_epochs + 1):
        student.train()
        running_loss = 0.0

        for _ in range(num_batches):
            inputs, targets, indices = curriculum.sample_batch(args.batch_size)
            inputs = inputs.to(device)
            targets = targets.to(device)
            indices = indices.to(device)

            logits = student(inputs)
            per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
            loss = curriculum.curriculum_loss(per_sample_loss, logits, indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            curriculum.update_after_batch(student)

        train_acc = evaluate_accuracy(student, train_dataset, device, args.batch_size)
        val_acc = evaluate_accuracy(student, val_dataset, device, args.batch_size)
        print(
            f"epoch={epoch:02d} "
            f"pool_size={curriculum.current_pool_size():4d}/{len(train_dataset)} "
            f"lambda_kl={curriculum.config.lambda_kl:.4f} "
            f"loss={running_loss / num_batches:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f}"
        )

    print()
    print(f"final_val_acc={evaluate_accuracy(student, val_dataset, device, args.batch_size):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Toy demo for Adaptive Curriculum Learning.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--teacher-epochs", type=int, default=18)
    parser.add_argument("--student-epochs", type=int, default=20)
    parser.add_argument("--teacher-lr", type=float, default=0.08)
    parser.add_argument("--student-lr", type=float, default=0.08)
    parser.add_argument("--pace-p", type=float, default=0.08)
    parser.add_argument("--pace-q", type=float, default=1.25)
    parser.add_argument("--pace-r", type=int, default=16)
    parser.add_argument("--inv", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=-0.05)
    parser.add_argument("--lambda-kl", type=float, default=0.05)
    parser.add_argument("--lambda-kl-decay", type=float, default=0.0)
    parser.add_argument("--lambda-kl-min", type=float, default=0.0)
    parser.add_argument("--difficulty-warmup-batches", type=int, default=0)
    parser.add_argument(
        "--score-mode",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "confidence"],
    )
    parser.add_argument("--disable-class-balance", action="store_true")
    return parser


if __name__ == "__main__":
    run_demo(build_parser().parse_args())
