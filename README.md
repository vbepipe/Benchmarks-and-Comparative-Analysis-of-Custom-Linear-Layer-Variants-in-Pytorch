# Benchmarks-and-Comparative-Analysis-of-Custom-Linear-Layer-Variants-in-Pytorch

Benchmarking and comparative analysis of custom PyTorch linear layer variants against the standard PyTorch nn.Linear. 

Includes results on MNIST and FashionMNIST datasets with detailed accuracy comparisons.


# Vinayak Benchmark Layers

This repository contains benchmarks of custom PyTorch linear layer variants developed and tested against the standard `nn.Linear` baseline.  
Experiments were conducted on **MNIST** and **FashionMNIST** datasets to measure accuracy, generalization, and stability.

---

## 📊 Unified Benchmark Comparison

| **Layer**              | **Dataset**     | **Peak Accuracy (%)** | **Final Accuracy (%)** | **Best Epoch** | **Notes** |
|-------------------------|-----------------|------------------------|-------------------------|----------------|-----------|
| **Baseline (Linear)**   | MNIST           | 99.36                 | 99.30                  | 10 / 17        | Smooth convergence, stable generalization. |
|                         | FashionMNIST    | 91.78                 | 91.69                  | 15             | Mild late overfitting. |
| **NewStandardLinear**   | MNIST           | 99.26                 | 99.24                  | 14             | Matches baseline, slightly less stable loss. |
|                         | FashionMNIST    | 91.99                 | 91.80                  | 16             | Similar to baseline, marginally better mid-epochs. |
| **VinayakPatelFast**    | MNIST           | 99.24                 | 99.21                  | 15             | Competitive, slightly noisier validation. |
|                         | FashionMNIST    | **92.20**             | 91.35                  | 29             | Best **peak accuracy**, but overfits late. |
| **VinayakPatelFast2**   | MNIST           | 99.31                 | 99.27                  | 16             | Balanced accuracy & stability. |
|                         | FashionMNIST    | 91.93                 | 91.52                  | 15             | Good generalization, avoids major collapse. |
| **VinayakPatelFast3**   | MNIST           | **99.38**             | 99.28                  | 14             | Best MNIST accuracy, slight mid-epoch instability. |
|                         | FashionMNIST    | 91.85                 | **91.86**              | 22             | Best **final accuracy**, resilient generalization. |

---

## 🧩 Key Insights

- On **MNIST**, all custom layers match the baseline; `Fast3` edges ahead with **99.38% peak**.
- On **FashionMNIST**:
  - `Fast` achieves the highest **peak (92.20%)**, though it overfits late.
  - `Fast3` achieves the best **final accuracy (91.86%)**, outperforming baseline.
  - `Fast2` provides the most **stable and predictable** convergence.

---

## 📂 Repository Contents

- `benchmarks/` → Training logs for each dataset and layer.
- `results/unified_results.md` → Unified benchmark table & analysis.
- `README.md` → Summary and quick insights.

---

## 🚀 Usage

This repo is intended for:
- Researchers comparing novel layer architectures.
- Practitioners evaluating trade-offs between peak accuracy, stability, and generalization.
- Students learning about benchmarking in PyTorch.

---

## 📌 Author

Developed by **Vinayak Patel** as part of ongoing experimentation in neural network architecture design.
