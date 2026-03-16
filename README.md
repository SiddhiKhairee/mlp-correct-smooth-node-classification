# Enhanced MLP + Correct & Smooth for Imbalanced Node Classification

This project explores improvements to the **MLP + Correct & Smooth (C&S)** framework for large-scale **node classification** on the **OGBN-Products** dataset from the Open Graph Benchmark.

The goal of this project is to improve model performance under **severe class imbalance** while maintaining computational efficiency.

---

# Project Overview

The **OGBN-Products** dataset represents an Amazon product co-purchasing network:

- **Nodes:** Amazon products  
- **Edges:** Products frequently purchased together  
- **Node Features:** 100-dimensional feature vectors  
- **Task:** Predict the product category  
- **Classes:** 47 categories  
- **Nodes:** ~2.4 million  

A major challenge in this dataset is **class imbalance**, where the largest class contains ~700k nodes while the smallest contains fewer than 10k nodes.

Traditional models tend to perform well on majority classes but poorly on minority classes.

This project modifies the **MLP + C&S baseline model** to address this issue.

---

# Baseline Model

The baseline used is **MLP + Correct & Smooth**, proposed in:

Huang et al., *Combining Label Propagation and Simple Models Out-performs Graph Neural Networks*  
https://arxiv.org/abs/2010.13993

Baseline pipeline:

1. Train an **MLP classifier** using node features.
2. Apply **Correct** step  
   - Propagates residual errors across the graph.
3. Apply **Smooth** step  
   - Diffuses predictions across neighboring nodes.

Despite its simplicity, this model ranks highly on the **OGB leaderboard** while using far fewer parameters than GNNs.

Baseline performance:

| Metric | Score |
|------|------|
| Validation Accuracy | 91.14% |
| Test Accuracy | 83.69% |

---

# Proposed Improvements

This project introduces **three main modifications** to the baseline model.

---

## 1. Weighted Random Sampling

The dataset exhibits a **70:1 imbalance ratio** between the largest and smallest classes.

To mitigate this:

- Compute **inverse frequency class weights**
- Sample training instances using **weighted random sampling**
- Oversample minority classes during training

Expected effects:

- Increased exposure to minority classes
- More balanced learning across all classes

Tradeoff:

- Slight decrease in overall accuracy due to reduced majority-class bias.

---

## 2. Ensemble of MLPs

Instead of using a single MLP, the model trains **three independent MLPs** with different random initializations.

Final predictions are obtained using **probability averaging**:

\[
P_{ensemble}(y|x) = \frac{1}{K} \sum_{k=1}^{K} P_k(y|x)
\]

Benefits:

- Reduces model variance
- Improves robustness
- Mitigates poor local minima

### MLP Architecture

- Input dimension: 100
- Hidden layers: 3
- Hidden size: 256
- Activation: ReLU
- Dropout: 0.5
- Output dimension: 47 classes

### Training Configuration

- Optimizer: Adam
- Learning rate: 0.01
- Batch size: 4096
- Epochs: 100

---

## 3. Iterative Correct & Smooth

The baseline applies **Correct → Smooth only once**.

This project proposes **multiple iterations**:

```
Correct → Smooth → Correct → Smooth
```

Motivation:

- Later smoothing steps create more confident predictions.
- These predictions can serve as **pseudo-labels** for further correction.

Expected benefits:

- Better long-range information propagation
- Progressive refinement of predictions

Configuration:

| Parameter | Value |
|------|------|
Correction α | 1.0 |
Smoothing α | 0.8 |
Propagation Steps | 50 |
Iterations | 2 |

---

# Final Pipeline

The complete training pipeline:

```
1. Load OGBN-Products dataset
2. Compute class weights
3. Create weighted sampler
4. Train ensemble of 3 MLP models
5. Average prediction probabilities
6. Apply iterative Correct & Smooth
7. Output final node predictions
```

Memory optimization:

- Models trained sequentially
- Predictions averaged before graph propagation
- Graph stored as sparse adjacency matrix

---

# Experimental Setup

Experiments were conducted using:

- **Kaggle GPU Runtime**
- GPU: NVIDIA Tesla P100 (16GB)
- CPU: Intel Xeon
- RAM: ~32GB

Spectral embeddings were excluded due to computational constraints for the 2.4M node graph.

---

# Results

| Metric | Baseline | Our Model |
|------|------|------|
| Training Accuracy | ~95% | **97.48%** |
| Validation Accuracy | 91.1% | 88.67% |
| Test Accuracy | 83.69% | 72.14% |

---

# Discussion

Although overall **test accuracy decreased**, the results highlight an important issue:

**Standard accuracy is a poor metric for imbalanced datasets.**

The baseline achieves high accuracy largely by predicting majority classes.

Our approach shifts the model's focus toward:

- Improved learning across all classes
- Better representation of minority classes
- More balanced training distribution

Training accuracy increased significantly, indicating improved learning across the full class distribution.

Future evaluations should include:

- Macro F1 Score
- Precision–Recall AUC
- Minority class recall

---

# Future Work

Potential extensions of this project include:

- Incorporating **spectral graph embeddings**
- Using **macro-F1 and PR-AUC** for evaluation
- Hyperparameter optimization using **Optuna**
- Testing larger architectures such as **GAMLP + RLU + SCR + C&S**
- Exploring alternative imbalance techniques such as **focal loss**

---

# How to run
To Run the code Follow the following Instructions
- Open up the ml-graphs-final-project in Kaggle (In Kaggle settings, go to Accelerator and GPU 100)
- Turn on the internet in the Kaggle settings
- In the dataset section, Add the zipped folder "CorrectAndSmooth-master-up"
- Run all the Kaggle cells

# References

Huang, Q., He, H., Singh, A., Lim, S., & Benson, A. R.  
*Combining Label Propagation and Simple Models Out-performs Graph Neural Networks*  
https://arxiv.org/abs/2010.13993

Open Graph Benchmark  
https://ogb.stanford.edu

OGBN-Products Dataset  
https://ogb.stanford.edu/docs/nodeprop/#ogbn-products

---

#Author
<table>
  <tr>
  <td align="center"><a href="https://github.com/SiddhiKhairee"><img src="https://avatars.githubusercontent.com/SiddhiKhairee" width="100px;" alt=""/><br /><b>Siddhi Khaire</b></a><br /></td>
  </tr>
</table>
