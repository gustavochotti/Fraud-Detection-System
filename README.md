# 💳 Credit Card Fraud Detection System

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Made with scikit-learn](https://img.shields.io/badge/ML-sklearn-informational)

End-to-end ML project to detect **fraudulent credit card transactions**. It covers **EDA → preprocessing → imbalanced learning (SMOTE) → model training/evaluation → real-time inference via Flask API**.

---

## 📋 Table of Contents
- [Project Description](#-project-description)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
  - [Train](#1-train)
  - [API Inference](#2-api-inference)
  - [Batch Inference (CSV)](#3-batch-inference-csv)
- [Results](#-results)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 📝 Project Description
This project builds a **binary classifier** to identify rare fraud cases in a highly imbalanced dataset. We use **SMOTE** to oversample minority class and optimize evaluation toward **Recall/AUC** to reduce dangerous **false negatives**. A **Flask REST API** serves real-time predictions.

---

## ✨ Features
- **EDA**: quick insights/visuals about transaction patterns.  
- **Preprocessing**: `StandardScaler` for numeric features.  
- **Imbalanced Learning**: **SMOTE** applied on training data.  
- **Model**: `RandomForestClassifier` as strong baseline for tabular data.  
- **Evaluation**: Confusion Matrix, Classification Report (Precision/Recall/F1), **ROC-AUC**.  
- **Serving**: **Flask API** for single-transaction inference.  
- **Batch**: CLI script to score CSV files.

---

## 🧰 Tech Stack
- **Python**, **scikit-learn**, **imblearn (SMOTE)**, **pandas**, **numpy**, **matplotlib**
- **Flask** for REST API

---

## 📁 Project Structure
```
Fraud-Detection-System/
│── .gitignore
│── LICENSE
│── README.md
│── requirements.txt
│
├── artifacts/              # Saved model & preprocessing objects
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── columns.json
│
├── data/                   # Put datasets here (ignored in VCS)
│   ├── creditcard.csv
│   └── novas_transacoes.csv (example input for batch)
│
├── notebooks/              # EDA & experiments
│   └── 1_analise_exploratoria.ipynb
│
├── reports/                # Plots/metrics exported by training
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── src/                    # Core project code
    ├── train_model.py
    ├── predict_api.py
    └── predict_from_csv.py
```

---

## 🚀 Getting Started

**Prerequisites**
- Python **3.9+**
- Git

**Installation**
## 1) Clone
```bash
git clone https://github.com/gustavochotti/Fraud-Detection-System.git
cd Fraud-Detection-System
```

## 2) Create & activate venv
```bash
python -m venv venv
```
### Linux/Mac
```bash
source venv/bin/activate
```
### Windows
```bash
.\venv\Scripts\activate
```
## 3) Install deps
```bash
pip install -r requirements.txt
```

**Dataset**
- Download from Kaggle and place as `data/creditcard.csv`  
  ➡ https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🛠️ Usage

### 1) Train
```bash
python src/train_model.py
```
Outputs go to `artifacts/` (model, scaler, columns) and `reports/` (plots).

### 2) API Inference
Start server:
```bash
python src/predict_api.py
```
Send a request (curl/Postman):
```bash
curl -X POST -H "Content-Type: application/json"   -d '{"Time": 406, "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99, "V14": -4.28, "Amount": 0.0}'   http://127.0.0.1:5000/predict
```
Response includes `prediction` and `fraud_probability`.

### 3) Batch Inference (CSV)
```bash
python src/predict_from_csv.py --filepath data/novas_transacoes.csv
```

---

## 📊 Results
- **Confusion Matrix**: strong Recall on fraud class (minimizing false negatives).  
- **Top Features** (model-dependent): often `V14`, `V4`, `V3`.  
See `reports/` for plots (add screenshots to this section for portfolio appeal).

---

## 🗺️ Roadmap
- Threshold tuning & probability calibration
- Experiment tracking (e.g., MLflow)
- Docker image & simple deploy (Render/Fly.io)
- Add unit tests for preprocessing/predict pipeline
- Add `/health` and `/version` endpoints to API

---

## 📄 License
MIT — see [LICENSE](LICENSE).
