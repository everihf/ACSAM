# Adaptive Curriculum Learning

This folder contains a clean, standalone reproduction of the core algorithm
from:

Yajing Kong, Liu Liu, Jun Wang, Dacheng Tao.
"Adaptive Curriculum Learning." ICCV 2021.

## Files

- `adaptive_curriculum.py`: paper-faithful ACL implementation.
- `toy_demo.py`: minimal runnable example on a synthetic classification task.
- `__init__.py`: package exports.

## What Is Reproduced

The implementation follows the main paper loop:

1. Initialize a pseudo-ideal difficulty score `s0` from a pretrained source.
2. Sort the training set by difficulty from easy to hard.
3. Build a sample pool with an exponential pacing rule.
4. Uniformly sample each mini-batch from the current pool.
5. Optimize:

   `L = CE + lambda * KL(student || teacher)`

6. Every `inv` mini-batches, update the difficulty score with:

   `s <- (1 - alpha) * s + alpha * s_cur`

## Notes

- The default score mode is `cross_entropy`, which matches the paper's loss-based
  description and theoretical analysis.
- If you already have external initial scores (for example, from an
  Inception-feature + SVM ranking pipeline), pass them in as
  `initial_difficulty`.
- Class-balanced pool selection is enabled by default when dataset labels can be
  inferred.
- The pacing rule includes the `batch_size / 100` scaling used in the common
  public ACL code path. This is exposed as `pacing_reference_batch_size` in the
  config so it can be changed explicitly.

## Quick Start

Run the toy demo:

```bash
python ACL/toy_demo.py
```

Run CIFAR training with `example/train.py`-style arguments:

```bash
python ACL/train.py \
  --seed 1 \
  --dataset cifar10 \
  --epochs 97 \
  --model resnet18 \
  --optimizer sgd \
  --curriculum_strategy adaptive \
  --pace_q 1.1 \
  --difficulty_warmup_batches 500 \
  --fixed_balance_order False \
  --adaptive_teacher_source teacher_model \
  --teacher_model resnet18 \
  --teacher_checkpoint example/checkpoints/04-22_18-42-45_teacher_model_cifar10-resnet-epoch200.pt
```

Disable the curriculum and train a plain baseline in the same entrypoint:

```bash
python ACL/train.py \
  --seed 1 \
  --dataset cifar10 \
  --epochs 97 \
  --model resnet18 \
  --optimizer sgd \
  --curriculum_strategy none
```

Outputs from `ACL/train.py` are written under:

- `ACL/logs`
- `ACL/metrics`
- `ACL/checkpoints`

Import the algorithm:

```python
from ACL import AdaptiveCurriculumConfig, AdaptiveCurriculumLearning

config = AdaptiveCurriculumConfig(
    pace_p=0.04,
    pace_q=1.1,
    pace_r=100,
    inv=50,
    alpha=-0.01,
    lambda_kl=0.01,
)

curriculum = AdaptiveCurriculumLearning(
    train_dataset=train_dataset,
    num_classes=num_classes,
    device="cuda",
    config=config,
    teacher_model=teacher_model,
)
curriculum.initialize(batch_size=128)

for _ in range(num_batches):
    inputs, targets, indices = curriculum.sample_batch(128)
    logits = model(inputs.to("cuda"))
    per_sample_loss = torch.nn.functional.cross_entropy(
        logits,
        targets.to("cuda"),
        reduction="none",
    )
    loss = curriculum.curriculum_loss(per_sample_loss, logits, indices.to("cuda"))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    curriculum.update_after_batch(model)
```
