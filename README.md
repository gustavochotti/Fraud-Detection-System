# Fraud-Detection-System

## 📝 Project Description

The goal of this project is to develop a machine learning model capable of identifying fraudulent credit card transactions among a large volume of legitimate ones. Given the highly imbalanced nature of the dataset, special techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are employed to ensure the model learns to effectively recognize the rare fraud cases. The final model is served through a Flask API for real-time predictions.

## ✨ Features

- **Exploratory Data Analysis (EDA):** In-depth analysis and visualization of the dataset's characteristics.
- **Data Preprocessing:** Feature scaling (`StandardScaler`) for numerical columns.
- **Imbalanced Data Handling:** SMOTE is applied to the training set to balance the class distribution.
- **Machine Learning Model:** A `RandomForestClassifier` is trained for its robustness and performance on tabular data.
- **Model Evaluation:** Detailed performance analysis using a Confusion Matrix, Classification Report (focusing on Precision and Recall), and AUC-ROC score.
- **API for Real-Time Predictions:** A Flask-based REST API to serve the model and provide on-demand fraud analysis.
- **Batch Prediction Script:** A command-line script to classify a batch of new transactions from a CSV file.

## 📁 Project Structure

The repository is organized with a clean and scalable structure:

└── 📂 projeto_deteccao_fraude/
├── 📄 .gitignore
├── 📄 LICENSE
├── 📄 README.md
├── 📄 requirements.txt
├── 📂 artifacts/
│   ├── columns.json
│   ├── fraud_model.pkl
│   └── scaler.pkl
├── 📂 data/
│   ├── creditcard.csv
│   └── novas_transacoes.csv
├── 📂 notebooks/
│   └── 1_analise_exploratoria.ipynb
└── 📂 src/
├── predict_api.py
├── predict_from_csv.py
└── train_model.py

---

## 🚀 Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```
Com certeza. Um README.md bem escrito é a porta de entrada do seu projeto no GitHub e é crucial para um bom portfólio.

Aqui está um template completo, totalmente em inglês, que você pode copiar e colar diretamente em um arquivo README.md na raiz do seu projeto. Ele explica o projeto, como instalá-lo (incluindo o link para o dataset) e como usar cada um dos seus componentes.

Conteúdo para o README.md
Markdown

# Credit Card Fraud Detection System

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An end-to-end machine learning project to build a robust system for detecting fraudulent credit card transactions. This repository covers the entire workflow from data analysis and preprocessing to model training, evaluation, and deployment via a REST API.

---

## 📋 Table of Contents
- [Project Description](#-project-description)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Making Predictions via API](#2-making-predictions-via-api)
  - [3. Making Batch Predictions from a CSV](#3-making-batch-predictions-from-a-csv)
- [Results](#-results)
- [License](#-license)

---

## 📝 Project Description

The goal of this project is to develop a machine learning model capable of identifying fraudulent credit card transactions among a large volume of legitimate ones. Given the highly imbalanced nature of the dataset, special techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are employed to ensure the model learns to effectively recognize the rare fraud cases. The final model is served through a Flask API for real-time predictions.

## ✨ Features

- **Exploratory Data Analysis (EDA):** In-depth analysis and visualization of the dataset's characteristics.
- **Data Preprocessing:** Feature scaling (`StandardScaler`) for numerical columns.
- **Imbalanced Data Handling:** SMOTE is applied to the training set to balance the class distribution.
- **Machine Learning Model:** A `RandomForestClassifier` is trained for its robustness and performance on tabular data.
- **Model Evaluation:** Detailed performance analysis using a Confusion Matrix, Classification Report (focusing on Precision and Recall), and AUC-ROC score.
- **API for Real-Time Predictions:** A Flask-based REST API to serve the model and provide on-demand fraud analysis.
- **Batch Prediction Script:** A command-line script to classify a batch of new transactions from a CSV file.

## 📁 Project Structure

The repository is organized with a clean and scalable structure:

└── 📂 project_deteccao_fraude/
├── 📄 .gitignore
├── 📄 https://www.google.com/search?q=LICENSE
├── 📄 README.md
├── 📄 requirements.txt
├── 📂 artifacts/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── columns.json
├── 📂 data/
│   └── creditcard.csv
├── 📂 notebooks/
│   └── 1_analise_exploratoria.ipynb
└── 📂 src/
├── train_model.py
└── predict_api.py


---
## 🚀 Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

- Python 3.9 or higher
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Create and activate a virtual environment:**
    *(This is a recommended best practice to isolate project dependencies.)*

    - For Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - For macOS / Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset:**
    The dataset used for training is too large to be hosted on GitHub directly. Please download it from the official source on Kaggle and place the `creditcard.csv` file inside the `data/` directory.

    * **Download Link:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🛠️ Usage

This project has three main functionalities: training the model, running the API, and making batch predictions.

### 1. Training the Model

To train the model from scratch, run the training script from the project's root directory. This script will process the data from the `data/` folder and save the final model, scaler, and column order into the `artifacts/` folder. It will also generate the evaluation plots in the `reports/` folder.

```bash
python src/train_model.py
```

### 2. Making Predictions via API
The API allows for real-time, single-transaction predictions.

# Step A: Start the API Server
Run the API script from the project's root directory. The server will start and wait for requests.

```bash
python src/predict_api.py
```

# Step B: Send a Prediction Request
Open a new terminal and use a tool like curl (or Postman/Insomnia) to send a POST request with the transaction data in JSON format.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"Time": 406, "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99, "V5": -0.52, "V6": -1.43, "V7": -2.54, "V8": 1.39, "V9": -2.77, "V10": -2.77, "V11": 3.2, "V12": -2.89, "V13": -0.59, "V14": -4.28, "V15": 0.38, "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42, "V20": -0.48, "V21": 0.52, "V22": 0.94, "V23": -0.15, "V24": -0.91, "V25": 0.0, "V26": -0.22, "V27": 0.08, "V28": -0.15, "Amount": 0.0}' [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict)
```

The API will return a JSON response with the prediction and the fraud probability.

### 3. Making Batch Predictions from a CSV
To classify a whole file of new transactions at once, use the predict_from_csv.py script.

```bash
python src/predict_from_csv.py --filepath data/teste_10_transacoes.csv
```
The script will print a table in the terminal with the prediction for each transaction in the specified CSV file.

## 📊 Results
The trained model demonstrates excellent performance on the test set, especially in identifying fraudulent transactions, which is the primary goal.

# Confusion Matrix
The confusion matrix shows that the model correctly identified 15 out of 16 fraud cases, resulting in a very low number of dangerous False Negatives.

# Feature Importance
The model relies heavily on a few key features to make its predictions, with V14, V4, and V3 being the most influential.

### 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.




























