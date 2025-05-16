# SPMI Reproducibility Study

> **Learning with Partial-Label and Unlabeled Data: A Uniform Treatment for Supervision Redundancy and Insufficiency**  
> *Liu et al.*, ICML 2024  
> ---------------------------  
> **This repository** contains our reproduction of SPMI, a mutual-information-based framework for semi-supervised partial-label learning. 

---

## Table of Contents

- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Implementation Details](#implementation-details)  
  - [Key Components](#key-components)  
  - [Ablation Study](#ablation-study)  
  - [Implementation Challenges](#implementation-challenges)  
- [Experimental Setup](#experimental-setup)  
  - [Datasets & Splits](#datasets--splits)  
  - [Hyperparameters & Augmentations](#hyperparameters--augmentations)  
- [Usage](#usage)  
  - [Running SPMI Experiments](#running-spmi-experiments)  
  - [Running Baseline (PRODEN+FixMatch)](#running-baseline-prodenfixmatch)  
- [Dependencies](#dependencies)  
---

## Overview

SPMI proposes a **unified** semi-supervised partial-label learning framework that:

1. **Expands** redundant partial-labels via a mutual information criterion.  
2. **Condenses** noisy candidate sets using a KL-divergence-based score.  
3. **Smoothly updates** class priors with EMA.

The authors claim SPMI outperforms composite baselines (PRODEN + FixMatch), but our reproduction struggled to match their 85–95% accuracies, revealing multiple ambiguities and potential pitfalls.

---

## Repository Structure
├── spmi.py              # Core SPMI algorithm  
├── model.py             # LeNet & WideResNet definitions  
├── dataset.py           # Data loaders & transforms  
├── train.py             # Training loop & utils  
├── ablation.py          # Ablation-study harness  
├── fminstexp.py         # Fashion-MNIST  
├── cifar10exp.py        # CIFAR-10  
├── cifar100test1.y      # CIFAR100  
├── cifar100test2.py     # CIFAR100  
├── SHVNexp.py           # SVHN  
├── ProdenFixmatch.py    # PRODEN+FixMatch baseline  
├── logs/                # Logs and Results
└── README.md            # This document  

## Implementation Details

### Key Components

- **Label Expansion** (`spmi.py`): Implements Eq. 11 to add probable labels based on mutual information.  
- **Label Condensation** (`spmi.py`): Applies Eq. 15 to prune the candidate set via an information score.  
- **Class Prior EMA** (`spmi.py`): Updates priors with Eq. 16; handles edge cases when α = 1.0.  

### Ablation Study

`ablation.py` lets you toggle:
- Initialization fallback  
- Label generation  
- Label condensation  

and measure their individual contributions on each dataset.

### Implementation Challenges

- **Initialization Fallback**: Added top-2 selection when no candidate meets threshold.  
- **Numerical Stability**: Inserted ε-smoothing in KL-divergence.  
- **Unused β Penalty**: Made the information-bottleneck term optional (it wasn’t used in reported experiments).  
- **Multi-GPU Issues**: DataParallel breaks the stateful candidate sets—**use a single GPU** for SPMI.

---

## Experimental Setup

### Datasets & Splits

| Dataset        | Labeled / Unlabeled | Partial-rate (p) |
| -------------- | ------------------- | ---------------- |
| Fashion-MNIST  | 1 000 / 4 000       | 0.3  / 0.3       |
| CIFAR-10       | 1 000 / 4 000       | 0.3  / 0.3       |
| CIFAR-100      | Not Conducted       |     -            |
| SVHN           | 1 000 /  —          | 0.3  /           |

### Hyperparameters & Augmentations

- **Batch size**: 256  
- **LR**: 0.03 with cosine schedule, warm-up 10 epochs  
- **Optimizer**: SGD (momentum = 0.9, weight_decay = 5e-4)  
- **Total epochs**: 500  
- **Augmentations**:  
  - Fashion-MNIST: RandomCrop → RandomHorizontalFlip → Cutout  
  - CIFAR/SVHN: RandomCrop → RandomHorizontalFlip → AutoAugment → Cutout  

---

## Usage

### Running SPMI Experiments

# Example: Fashion-MNIST
python fminstexp.py

# CIFAR-10
python cifar10exp.py

# SVHN
python SHVNexp.py

# Ablation study (e.g. for Fashion-MNIST, 500 epochs)
python ablation.py --dataset fmnist --epochs 500

# For Fixmatch bash script with output redirect (output redir is in the bash file)
bash run_proden_experiments.sh

## Dependencies
run requirements.txt

pip install -r requirements.txt





