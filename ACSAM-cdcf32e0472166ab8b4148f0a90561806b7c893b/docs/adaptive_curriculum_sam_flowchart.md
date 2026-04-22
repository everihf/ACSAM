# Algorithm 1 自适应课程锐度感知最小化算法（Adaptive Curriculum + SAM）

> 说明：按你给的示意图风格，使用“算法步骤”格式；并显式补上 `curriculum_finished` 前后两套训练分支。

```text
1:  初始化：
    随机初始化学生模型参数 θ₀；
    用教师模型计算全训练集初始难度分数 dᵢ⁰；
    设置超参数：ρ, η, α, λ₁, pace_p, pace_q, pace_r, inv, warmup, T。

2:  for epoch = 1 to T do
3:      if curriculum_finished = False then
4:          课程批次构建：
            根据当前难度分数 {dᵢ} 对样本排序（由易到难），
            计算课程规模
                epoch_size = N · min(pace_p · pace_q^(⌊batch·(bs/100)/pace_r⌋), 1)，
            构建课程子集 DataLoader（仅前 epoch_size 个样本）。
5:      else
6:          课程结束：
            直接构建全数据集 DataLoader（不再按难度截断样本）。
7:      end if

8:      for 每个批次 Bₘ do
9:          步骤一：定义当前批次损失
10:             先计算平滑交叉熵逐样本损失：
                    lᵢ^sce ← smooth_crossentropy(f_θ(xᵢ), yᵢ)
11:             if curriculum_finished = False then
12:                 使用课程期损失（监督 + 蒸馏）：
                    L_B(θ) = mean(l^sce) + λ₁ · KL(p_s(θ) || p_t)
13:             else
14:                 使用常规损失：
                    L_B(θ) = mean(l^sce)
15:             end if

16:         步骤二：SAM 两步更新
17:             一阶梯度：g ← ∇_θ L_B(θ)
18:             扰动：ε ← ρ · g / ||g||₂
19:             扰动点梯度：g_SAM ← ∇_θ L_B(θ + ε)
20:             参数更新：θ ← θ − η · g_SAM

21:         步骤三：更新课程状态（每个 batch 后）
22:             更新全局 batch 计数与 epoch 内状态。
23:             if (batch % inv == 0) and (batch > warmup) and (curriculum_finished = False) then
24:                 重估训练集样本损失 lᵢ，并做 EMA：
                        dᵢ ← (1 - α)·dᵢ + α·lᵢ
25:                 可选：λ₁ ← max(bottom_λ₁, λ₁ - λ₁_decay)
26:             end if
27:             if epoch_size >= N then
28:                 curriculum_finished ← True
29:             end if
30:      end for
31:  end for
```

## 关键点（你指出的补充项）

- **课程结束前（`curriculum_finished=False`）**：
  - 数据：按难度排序后，只取课程子集训练；
  - 损失：`smooth_crossentropy` + 蒸馏 `KL`（课程损失）。
- **课程结束后（`curriculum_finished=True`）**：
  - 数据：直接切换为全训练集；
  - 损失：仅 `smooth_crossentropy`（不再叠加课程蒸馏项）。

## 与代码实现的对应关系

- `curriculum_finished` 分支与全量 DataLoader 切换：`example/train.py` 的训练循环分支。
- 课程期损失 vs 课程结束后损失：`example/train.py` 中
  `if curriculum is not None and not curriculum.curriculum_finished`。
- 课程规模扩张与样本选择：`curriculum/adaptive_algorithms.py` 的 `data_curriculum`。
- 难度 EMA 更新：`curriculum/adaptive_algorithms.py` 的 `_difficulty_measurer`。
- SAM 两步更新：`sam.py` 的 `first_step` / `second_step`。
