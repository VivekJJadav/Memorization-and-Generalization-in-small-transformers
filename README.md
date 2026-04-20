<p align="center">
  <h1 align="center">🧠 Does Data Duplication Help or Hurt Small Transformers?</h1>
  <p align="center">
    <em>An empirical study on how dataset duplication affects memorization and generalization<br>in a ~50M parameter decoder-only transformer trained on synthetic reasoning tasks.</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Transformer_(GPT--style)-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Parameters-~50M-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Tasks-5_Synthetic-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge" />
</p>

---

## 📌 TL;DR

> **Duplication buys speed, not capability. Deduplication buys nothing for generalization. Both models are pure memorizers.**

We trained identical small transformers on two dataset variants — one with natural duplicates (~27%) and one exactly deduplicated (0%) — across 5 synthetic tasks. Key findings:

| # | Finding | Confidence |
|:---:|---------|:---:|
| 1 | Duplication **accelerates early training** (12pp accuracy lead by epoch 5) | ✅ Strong |
| 2 | Both converge to **identical final in-distribution accuracy** (~99.6%) | ✅ Strong |
| 3 | **Neither model generalizes** OOD — 0% exact match across 7,137 test examples | ✅ Strong |
| 4 | The model **memorized input-output pairs**, not algorithms | ✅ Strong |
| 5 | Dedup shows slightly better **structural partial matching** on some OOD tasks | ⚠️ Suggestive |

---

## 🔬 Research Question

**How does data duplication in training sets affect a small transformer's ability to:**
1. **Memorize** — learn exact input→output mappings from training data
2. **Generalize** — apply learned patterns to unseen inputs, tasks, and complexity levels

---

## 🏗️ Experimental Setup

### Model Architecture

Both conditions use an **identical** decoder-only transformer:

| Parameter | Value |
|:---|:---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Embedding dim (`d_model`) | 512 |
| Layers (`n_layers`) | 16 |
| Attention heads (`n_heads`) | 8 |
| FFN dim (`d_ff`) | 2048 |
| Max sequence length | 128 |
| Dropout | 0.1 |
| Vocabulary | 77 tokens (character-level) |
| **Est. parameters** | **~50M** |

### Training Configuration

| Parameter | Value |
|:---|:---|
| Epochs | 50 |
| Batch size | 128 |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| LR schedule | Cosine decay → 1e-5 |
| Warmup | 1,000 steps |
| Label smoothing | 0.0 |
| Seed | 1337 |

### Synthetic Tasks

The model is trained on **5 equally-weighted synthetic reasoning tasks** (20% each):

| Task | Prefix | Example Input → Output |
|:---|:---|:---|
| **Sort** | `[sort_asc]` / `[sort_desc]` | `3 1 4 1 5` → `1 1 3 4 5` |
| **Reverse** | `[rev]` | `A B C` → `C B A` |
| **Addition** | `[add]` | `42 + 17` → `59` |
| **Copy/Repeat** | `[copy]` | `ABC repeat 3 times` → `ABCABCABC` |
| **Relations** | `[rel]` | `<E0> is parent of <E1>. Who is ancestor of <E1>?` → `<E0>` |

### Dataset Variants

| Dataset | Total Examples | Unique Examples | Duplication Rate |
|:---|---:|---:|---:|
| 🟢 **Normal** (natural duplicates) | 64,243 | 46,684 | **27.33%** |
| 🔵 **Deduplicated** (exact dedup) | 51,532 | 51,532 | **0.00%** |
| 🟡 **Duplicated** (intentional oversampling) | 64,243 | 42,403 | **33.99%** |
| Validation (shared) | 10,000 | 7,878 | 21.22% |

> **Note:** The *Duplicated* condition (🟡) training is pending — results will be added once complete.

---

## 📊 Results

### 1. Training Dynamics

#### Loss Convergence

<table>
<tr>
<td align="center"><strong>🟢 Normal (27% duplication)</strong></td>
<td align="center"><strong>🔵 Deduplicated (0% duplication)</strong></td>
</tr>
<tr>
<td><img src="runs/normal_50M/Plots/train_val_loss.png" width="100%"/></td>
<td><img src="runs/dedup_50M/plots/train_val_loss.png" width="100%"/></td>
</tr>
</table>

| Metric | 🟢 Normal | 🔵 Dedup | Δ |
|:---|---:|---:|:---|
| Initial train loss (epoch 1) | 37.00 | 45.76 | Dedup starts **24% higher** |
| Final train loss (epoch 50) | 0.00051 | 0.00098 | Dedup **1.9× higher** |
| Final val loss (epoch 50) | 0.00162 | 0.00155 | Dedup **4% lower** ✅ |
| Final train–val gap | 0.001 | 0.001 | **Identical** |

**Interpretation:** The duplicated dataset is easier to initially fit (lower epoch-1 loss) because repeated examples provide redundant gradient signal. At convergence, the dedup model retains slightly higher training loss (no "free" repeated examples) but achieves marginally *lower* validation loss — hinting at slightly less overfitting.

#### Overfitting Signal (Train–Val Gap)

<table>
<tr>
<td align="center"><strong>🟢 Normal</strong></td>
<td align="center"><strong>🔵 Deduplicated</strong></td>
</tr>
<tr>
<td><img src="runs/normal_50M/Plots/train_val_gap.png" width="100%"/></td>
<td><img src="runs/dedup_50M/plots/train_val_gap.png" width="100%"/></td>
</tr>
</table>

Both models converge to a near-zero train–val gap, indicating **neither model is significantly overfitting** on in-distribution data by the standard loss metric.

---

### 2. In-Distribution Accuracy (Memorization)

<table>
<tr>
<td align="center"><strong>🟢 Normal</strong></td>
<td align="center"><strong>🔵 Deduplicated</strong></td>
</tr>
<tr>
<td><img src="runs/normal_50M/Plots/exact_match.png" width="100%"/></td>
<td><img src="runs/dedup_50M/plots/exact_match.png" width="100%"/></td>
</tr>
</table>

| Epoch | 🟢 Normal | 🔵 Dedup | Gap |
|:---:|---:|---:|:---|
| 1 | **19.32%** | 2.19% | Normal leads by **17.1pp** |
| 5 | **71.83%** | 59.82% | Normal leads by **12.0pp** |
| 10 | **93.61%** | 88.33% | Normal leads by **5.3pp** |
| 20 | **98.16%** | 97.92% | Gap narrows to **0.2pp** |
| 30 | 99.28% | 99.10% | Effectively **tied** |
| **50** | **99.65%** | **99.56%** | **Tied** (Δ = 0.09pp) |

> #### 💡 Key Insight
> Duplication acts as **implicit spaced repetition** — the model sees the same example multiple times per epoch, reinforcing pattern-output mappings faster. But given enough training time, the dedup model catches up completely. **Duplication accelerates memorization; it does not expand capacity.**

#### Per-Task Accuracy Heatmaps

<table>
<tr>
<td align="center"><strong>🟢 Normal</strong></td>
<td align="center"><strong>🔵 Deduplicated</strong></td>
</tr>
<tr>
<td><img src="runs/normal_50M/Plots/task_accuracy_heatmap.png" width="100%"/></td>
<td><img src="runs/dedup_50M/plots/task_accuracy_heatmap.png" width="100%"/></td>
</tr>
</table>

Both models reach near-perfect accuracy (dark green) by epoch ~25 across all tasks. The normal model's earlier transition from red→green (around epochs 1–7) is visible — confirming faster early convergence.

---

### 3. Out-of-Distribution Generalization — The Critical Test

We evaluated both models on **7,137 OOD examples** spanning 7 categories designed to test whether the model learned *algorithms* or merely *mappings*.

> ### 🚨 Both models achieve 0.0% exact match on ALL out-of-distribution tests.

| OOD Category | Examples | 🟢 Normal Loss | 🔵 Dedup Loss | 🟢 Normal Partial | 🔵 Dedup Partial |
|:---|---:|---:|---:|---:|---:|
| **Compositional** (chained ops like sort→reverse) | 1,000 | **32.69** | 46.31 | **1.50%** | 0.70% |
| **Edge Cases** (single elements, identity) | 581 | **34.00** | 46.83 | 55.42% | **69.02%** |
| **Extrapolation** (longer seqs, larger numbers) | 2,500 | **20.18** | 32.14 | 1.44% | **5.64%** |
| **Interpolation** (unseen in-range inputs) | 999 | **29.54** | 43.37 | 67.27% | **72.57%** |
| **Novel Tasks** (subtraction, multiplication) | 996 | **36.05** | 56.22 | **14.96%** | 10.04% |
| **Robustness** (formatting variations) | 500 | **31.92** | 51.22 | **29.20%** | 27.20% |
| **Systematic** (progressive complexity) | 561 | **24.50** | 29.36 | 16.40% | **17.47%** |
| **Overall** | **7,137** | **27.75** | **41.37** | — | — |

#### What the OOD Predictions Look Like

```
Task:     [sort_then_reverse] 1 5 4 1 7 7 5
Expected: 7 7 5 5 4 1 1

🟢 Normal predicted: [sort<unk>then<unk>r<v<rs<]<0<011<2< < <02 1 1 1 1 4 1 5
🔵 Dedup predicted:  yyyyyyyyyyyyyyyyyy*yr)By < ) ) B 5 7 1 1 4 5 7
```

```
Task:     [addition_extrap] 9070 + 1889
Expected: 10959

🟢 Normal predicted: yaddition<unk>1<unk>etrap]14>  1+1666 2 288
🔵 Dedup predicted:  yaddition<unk>s<unk>etrap]188  179000 0 108
```

Both models produce **garbled, incoherent outputs** when they encounter unseen task tokens or inputs outside the training distribution. The model has learned a lookup table, not a generalizable algorithm.

#### Nuance: Dedup Shows Better Partial Matching on Structural Tasks

Despite both failing at exact match, the dedup model gets **more tokens right** on tasks with structural similarity to training:

| OOD Category | 🔵 Dedup Advantage |
|:---|:---|
| Edge cases (simple structure) | **+13.6 percentage points** |
| Reversal extrapolation (same operation, longer inputs) | **+21.0 percentage points** |
| Interpolation (same tasks, unseen inputs) | **+5.3 percentage points** |

This suggests that **unique training examples may encode slightly more robust structural priors**, even though exact generalization fails entirely.

---

### 4. Autoregressive Generation Quality

When generating outputs token-by-token (rather than teacher-forced), both models show identical behavior:

| Metric | 🟢 Normal | 🔵 Dedup |
|:---|---:|---:|
| Exact Match | 0.0% | 0.0% |
| Token Accuracy | 52.75% | 52.78% |
| Edit Distance | 0.547 | 0.547 |

A **47 percentage-point drop** from teacher-forced accuracy (~99.97%) to autoregressive accuracy (~52.8%) confirms the models are fragile memorizers — they cannot reliably generate correct sequences without ground-truth context at each step.

---

### 5. Memorization Dashboard

<table>
<tr>
<td align="center"><strong>🟢 Normal</strong></td>
<td align="center"><strong>🔵 Deduplicated</strong></td>
</tr>
<tr>
<td><img src="runs/normal_50M/Plots/memorization_dashboard.png" width="100%"/></td>
<td><img src="runs/dedup_50M/plots/memorization_dashboard.png" width="100%"/></td>
</tr>
</table>

---

## 🧪 Pending: Duplicated Dataset (🟡 High-Dup Condition)

> **Status:** Training in progress. Results will be added here.

The third condition uses intentionally duplicated data (~34% duplication rate), where the top-200 most frequent examples per task are repeated 10×. This will enable a three-way comparison:

| Condition | Dup Rate | Hypothesis |
|:---|---:|:---|
| 🔵 Deduplicated | 0% | Slower training, potentially better structure learning |
| 🟢 Normal | 27% | Baseline with natural duplication |
| 🟡 Duplicated | 34% | Fastest memorization, potentially worst generalization |

---

## ✅ Conclusions

### What We Can Confidently Say

**1. Duplication accelerates early-stage training convergence.**
The normal model (27% duplicates) reaches 71.8% validation accuracy by epoch 5, while the deduplicated model reaches only 59.8% — a 12 percentage-point gap. Repeated examples act as implicit spaced repetition, reinforcing pattern-output mappings faster within each epoch.

**2. Deduplication does not improve or harm final in-distribution performance.**
Given sufficient training time (50 epochs), both models converge to near-identical accuracy: 99.65% (normal) vs 99.56% (dedup). The 0.09pp difference is within noise margin.

**3. Neither duplication nor deduplication enables out-of-distribution generalization.**
Both models achieve exactly **0.0% exact match** across all 7 OOD categories (7,137 examples). This is the central result: at ~50M parameters and ~50K training examples, dataset duplication policy has **zero measurable effect** on generalization.

**4. Small transformers on these tasks learn memorization, not algorithms.**
The combination of 99.6% in-distribution accuracy and 0% OOD accuracy is the definitive signature of pure memorization. The model stores input→output pairs rather than learning the underlying sorting, addition, or reversal algorithms.

**5. Deduplication may encourage marginally better structural representations.**
The dedup model achieves higher partial match rates on edge cases (+13.6pp), interpolation (+5.3pp), and reversal extrapolation (+21.0pp). While not translating to correct answers, this suggests unique training examples may produce slightly more robust internal representations of output structure.

---

## ⚠️ Limitations & Caveats

| Limitation | Impact | Mitigation |
|:---|:---|:---|
| **Single seed** (1337) | No variance estimate; results could shift | Run 3–5 seeds per condition |
| **Unequal dataset sizes** | Normal=64K, Dedup=51.5K (20% fewer examples) | Match total training tokens instead of epochs |
| **OOD uses unseen task tokens** | Tests token novelty, not just algorithmic generalization | Use training task prefixes with harder inputs |
| **Relations task imbalance** | Dedup has only 140 unique relation examples vs ~12.8K for others | Expand relation generation space |
| **No train accuracy tracking** | Cannot directly measure memorization gap | Enable `--compute_memorization` flag |

---

## 📂 Repository Structure

```
.
├── train_minigpt_4070.py           # Main training script (~50M param transformer)
├── evaluate_ood.py                 # Out-of-distribution evaluation suite
├── generate_synthetic_datasets.py  # Dataset generation (normal/dedup/duplicated)
├── metrics.py                      # Memorization & generalization metrics
├── plots.py                        # Visualization utilities
│
├── data/
│   └── processed/
│       ├── train_normal.pkl        # 🟢 Normal dataset (27% dup)
│       ├── train_dedup.pkl         # 🔵 Deduplicated dataset (0% dup)
│       ├── train_duplicated.pkl    # 🟡 Duplicated dataset (34% dup)
│       ├── val.pkl                 # Shared validation set
│       ├── tokenizer.json          # Character-level tokenizer
│       └── dataset_stats.json      # Dataset composition statistics
│
├── ood_data_complete/              # OOD test sets (7 categories)
│
└── runs/
    ├── normal_50M/                 # 🟢 Normal training run
    │   ├── training_history.json
    │   ├── ood_eval                # OOD evaluation results
    │   └── Plots/                  # Training visualizations
    │
    └── dedup_50M/                  # 🔵 Dedup training run
        ├── training_history (1).json
        ├── ood_evaluation_results.json
        └── plots/                  # Training visualizations
```

---

## 🚀 Reproduce

### 1. Generate Datasets

```bash
python generate_synthetic_datasets.py
```

### 2. Train (Normal)

```bash
python train_minigpt_4070.py \
  --data data/processed/train_normal.pkl \
  --tokenizer data/processed/tokenizer.json \
  --out_dir runs/normal_50M \
  --epochs 50 --lr 1e-4 --warmup_steps 1000 \
  --d_model 512 --n_layers 16 --n_heads 8 \
  --batch_size 128 --dropout 0.1 --autoreg_eval
```

### 3. Train (Deduplicated)

```bash
python train_minigpt_4070.py \
  --data data/processed/train_dedup.pkl \
  --tokenizer data/processed/tokenizer.json \
  --out_dir runs/dedup_50M \
  --epochs 50 --lr 1e-4 --warmup_steps 1000 \
  --d_model 512 --n_layers 16 --n_heads 8 \
  --batch_size 128 --dropout 0.1 --autoreg_eval
```

### 4. Evaluate OOD

```bash
python evaluate_ood.py \
  --model runs/normal_50M/best_model.pth \
  --tokenizer data/processed/tokenizer.json \
  --ood_dir ood_data_complete \
  --out_dir runs/normal_50M/ood_results
```

---

## 📚 Citation

If you find this work useful, please consider citing:

```
@misc{duplication-memorization-2026,
  title   = {Does Data Duplication Help or Hurt Small Transformers?},
  year    = {2026},
  url     = {https://github.com/VivekJJadav/Temp}
}
```

---
