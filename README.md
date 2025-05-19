# Role-Playing Steering with Sparse Autoencoders (SAE)

This project explores **role-specific behavior steering** in Large Language Models (LLMs) using features extracted by a **Sparse Autoencoder (SAE)**. By leveraging contrastive activation patterns between role-play and non-role-play inputs, our method systematically improves reasoning performance in both commonsense and arithmetic domains, while offering strong interpretability.

---

## 🔍 Method Overview

We construct **contrastive input pairs**, where each pair includes:

- One sample with a **role-playing prompt**
- One sample without a role-playing prompt

By comparing their activation differences, we use a pretrained **Sparse Autoencoder (SAE)** to extract the **top-k most discriminative features**. These features are then **injected into the residual stream** of the model, guiding it toward role-specific behavior and improving reasoning ability.

---

## 🧪 Experimental Setup

We evaluate our method across **three models**, **three datasets**, and **three evaluation settings**:

### 🧠 Models
- `Llama3.1-8B`
- `Gemma2-2B`
- `Gemma2-9B`

### 📚 Datasets
- **GSM8K** – arithmetic reasoning
- **SVAMP** – arithmetic reasoning
- **CSQA** – commonsense reasoning

### 🧾 Evaluation Settings
- **Zero-shot-CoT**
- **One-shot-CoT**
- **Few-shot-CoT**

---

## 📈 Results

Extensive experiments show that our **SAE-based steering method**:
- **Consistently improves performance** across models and tasks
- **Outperforms or matches prompt-based role-playing**
- Offers **greater stability and interpretability** by exposing and controlling internal feature activations

---

## 📁 Code Structure

├── data/
│ ├── train.csv # SVAMP training data
│ └── SVAMP.json # SVAMP test data
├── requirements.txt # Dependencies for running the code
└── sae_pipeline.py # Main script to run evaluations with and without steering

---

## ▶️ Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt

### 1. Install dependencies

```bash
python sae_pipeline.py
