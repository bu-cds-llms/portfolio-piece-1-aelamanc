[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/brKTKdOU)
# Portfolio Piece Assignment

# Neural Network Design Choices Using Fashion-MNIST

**Portfolio Piece 1** — Extending Lab 2 (Neural Network Exploration)

## Overview

In Lab 2, we built simple neural networks on XOR and MNIST where even basic MLPs hit 97%+, leaving little room to see how design choices matter. This project switches to **Fashion-MNIST** — same 28×28 grayscale format but a much harder 10-class task (t-shirts vs. pullovers, sneakers vs. ankle boots) — and systematically isolates individual variables to measure their real impact on learning.

## Methods

All experiments use PyTorch MLPs on Fashion-MNIST (60K train / 10K test, 10 classes). Each experiment changes exactly one variable from a common baseline to enable fair comparison.

**Architecture experiments:**
- **Depth** — 1, 2, 3, and 5 hidden layers (all width 128)
- **Width** — 64, 128, and 256 neurons (all 2 hidden layers)
- **Activation functions** — ReLU, LeakyReLU, GELU, and Sigmoid (2 layers, width 128)

**Optimization experiments:**
- **Optimizers** — SGD, SGD + Momentum, Adam, and AdamW
- **Learning rate schedules** — Constant LR, StepLR, and Cosine Annealing
- **Regularization** — Dropout (0.3), Weight Decay (L2), Batch Normalization, and all three combined

## Key Results

- **Depth:** Minimal impact — 1 to 5 hidden layers spanned only 0.5% (89.0% → 89.5%), with all configurations overfitting after epoch ~10.
- **Width:** Diminishing returns — a 4× parameter increase (64 → 256) yielded just 0.7% improvement.
- **Activations:** LeakyReLU led at 89.5%, Sigmoid trailed at 88.5% due to vanishing gradients; the 1% spread suggests activation choice is a minor factor for shallow MLPs.
- **Optimizers:** Adam and AdamW reached ~88% within 5 epochs while vanilla SGD lagged at ~83%, though all converged near 89% by epoch 25.
- **LR Schedules:** Small but consistent boost (~0.4%), with StepLR slightly outperforming Cosine Annealing and Constant LR.
- **Regularization:** Combining dropout, batch normalization, and weight decay was most effective at reducing the generalization gap.

## Repository Structure

```
portfolio-piece/
├── README.md
├── requirements.txt
├── notebooks/
│   └── main_analysis.ipynb
├── src/
│   └── utils.py                 # Training, evaluation, and visualization utilities
├── outputs/                     # Generated figures
└── data/                        # Fashion-MNIST (auto-downloaded, gitignored)
```

## How to Run

```bash
# 1. Clone the repo
git clone <repo-url>
cd portfolio-piece

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
cd notebooks
jupyter notebook main_analysis.ipynb
```

Fashion-MNIST downloads automatically via `torchvision` on first run (~30MB). The notebook takes approximately 15–25 minutes on CPU or ~5 minutes with GPU/MPS.

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- torchvision >= 0.15
- numpy, matplotlib, seaborn, scikit-learn, tqdm

See `requirements.txt` for pinned versions.