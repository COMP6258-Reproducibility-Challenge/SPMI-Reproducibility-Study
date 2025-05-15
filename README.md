# SPMI Reproducibility Study

> **Learning with Partial-Label and Unlabeled Data: A Uniform Treatment for Supervision Redundancy and Insufficiency**  
> *Liu et al.*, ICML 2024  
> ---------------------------  
> **This repository** contains our reproduction of SPMI, a mutual-information-based framework for semi-supervised partial-label learning. We rigorously re-implement, debug, and benchmark the method as part of the COMP6258 Reproducibility Challenge.

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
- [Results & Reproducibility Analysis](#results--reproducibility-analysis)  
  - [Key Findings](#key-findings)  
  - [What We Learned](#what-we-learned)  
  - [Recommendations](#recommendations)  
- [Dependencies](#dependencies)  
- [Citation](#citation)  
- [License](#license)  

---

## Overview

SPMI proposes a **unified** semi-supervised partial-label learning framework that:

1. **Expands** redundant partial-labels via a mutual information criterion.  
2. **Condenses** noisy candidate sets using a KL-divergence-based score.  
3. **Smoothly updates** class priors with EMA.

The authors claim SPMI outperforms composite baselines (PRODEN + FixMatch), but our reproduction struggled to match their 85–95% accuracies, revealing multiple ambiguities and potential pitfalls.

---

## Repository Structure

