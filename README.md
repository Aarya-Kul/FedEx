# 🧠 Federated CNN with Spectral Clustering (FedEx) on MNIST

This project explores challenges in federated learning (FL), specifically the **personalization gap** and **performance degradation** in non-IID data settings — where data distributions vary widely across clients. To address these limitations in classical Federated Averaging (FedAvg), we propose a novel extension called **FedEx**, which uses **spectral clustering** to group similar clients before aggregation, yielding more personalized and efficient learning.

We evaluate our method using a custom Convolutional Neural Network (CNN) trained on MNIST, where both IID and non-IID splits are simulated, and clustered training is applied across groups.

The Research Paper: [FedEx: An Exploration of Clustering Clients in Federated Learning](https://github.com/Aarya-Kul/FedEx/blob/main/FedEx_Research_Paper.pdf)
---

## 💡 Motivation

Federated Learning allows distributed clients to collaboratively train a global model without sharing their raw data. However, **FedAvg** — the standard FL algorithm — performs poorly under **non-IID settings**, leading to:

- Slower convergence
- Local models that diverge from the global optimum
- High personalization loss (accuracy drops on specific clients)

To combat this, we introduce **FedEx** — an enhanced FL algorithm that clusters clients based on the **cosine similarity** of their model updates (weights), using **Spectral Clustering**. Instead of a single global model, each cluster gets its own personalized aggregator.

---

## 🚀 What We Built

- 🧠 **FedAvg Baseline**: Standard Federated Averaging algorithm across clients
- 🧩 **FedEx (Clustering Extension)**:
  - Warm-up phase to collect model updates
  - Cosine similarity matrix calculation
  - Spectral Clustering to group clients
  - Cluster-specific model aggregation

- 🧵 **Threaded Client Simulation** with condition variables
- 🗂️ **IID & non-IID Partitioning** of MNIST via shard-based slicing
- 📉 **Global accuracy and convergence logging**

---

## 🧠 FedEx: Federated Clustering via Spectral Methods

**FedEx Pipeline**:
1. Clients train for 1 warm-up round locally (using FedAvg)
2. Server collects their weight vectors
3. Cosine similarity matrix is computed
4. Spectral clustering groups similar clients
5. Each cluster receives a separate server to aggregate its local models

**Why Spectral Clustering?**
- Does not require convexity assumptions
- Captures nonlinear group structures
- Robust under noisy similarity measures

---

## ⚙️ How to Run

### Install dependencies

```bash
pip install torch torchvision scikit-learn numpy
```

### Run Federated Learning

**FedAvg Baseline** (IID split by default):

```bash
python main.py 20
```

**FedEx with 3 Clusters**:

```bash
python main.py 20 --clusters 3
```

Optional args:
- `--iid`: switch between IID and non-IID data
- `--clients 20`: set number of clients
- `--epochs 5`: local training rounds per client per round

---

## 📊 Model Architecture

Custom CNN (`MNISTCNN`) includes:

- 2 Conv2D layers: 1→32 and 32→64 filters
- MaxPooling and ReLU
- 2 Fully Connected layers: 1024 → 512 → 10
- Softmax output for 10 MNIST classes

Training:
- Optimizer: Adam
- Loss: CrossEntropyLoss

---

## 📈 Results Summary

| Setting        | Algorithm | Accuracy | Notes                      |
|----------------|-----------|----------|----------------------------|
| IID            | FedAvg    | 98.5%    | Fast convergence           |
| Non-IID (split)| FedAvg    | 92.6%    | Slower convergence         |
| Non-IID        | FedEx (k=3)| **95.1%**| Better personalization, faster convergence |

---

## 📌 Future Work

- Apply FedEx to more complex datasets (CIFAR-10, FEMNIST)
- Explore Graph Neural Networks for similarity clustering
- Introduce dropout/regularization strategies for generalization
- Compare against other personalized FL methods (e.g. FedProx, Per-FedAvg)

---

**Contributors**: Aarya Kulshrestha
