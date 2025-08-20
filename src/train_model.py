# -*- coding: utf-8 -*-
"""
Script Completo para Treinamento de Modelo de Detecção de Fraude

Este script realiza o ciclo completo de um projeto de Machine Learning padrão:
1. Carregamento e Análise Exploratória dos Dados (EDA).
2. Pré-processamento dos dados (normalização).
3. Divisão em conjuntos de treino e teste de forma estratificada.
4. Tratamento do desbalanceamento de classes com a técnica SMOTE.
5. Treinamento de um modelo de classificação (Random Forest).
6. Avaliação detalhada do modelo com métricas apropriadas (Recall, Precision, Matriz de Confusão).
7. Visualização dos resultados para melhor interpretação.
8. Salvamento do modelo treinado e do normalizador para uso futuro (produção).

Autor: [Seu Nome Aqui]
Data: 19 de agosto de 2025
"""

# =============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =============================================================================
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

print("--- Início do Script de Treinamento de Modelo de Fraude ---")

# =============================================================================
# 2. CARREGAMENTO DOS DADOS
# =============================================================================
data_path = os.path.join('data', 'creditcard.csv')

try:
    df = pd.read_csv(data_path)
    print(f"Dataset '{data_path}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: O arquivo '{data_path}' não foi encontrado.")
    print("Certifique-se de que o dataset está na pasta 'data/'.")
    exit()

# =============================================================================
# 2.1 VERIFICAÇÃO E LIMPEZA DE DADOS FALTANTES (NaN)
# =============================================================================
print("\n--- Verificando e Removendo Linhas com NaN na Coluna 'Class' ---")

# Identifica os índices das linhas com NaN na coluna 'Class' ANTES de remover
indices_para_remover = df[df['Class'].isnull()].index

# Verifica se de fato há alguma linha para remover
if not indices_para_remover.empty:
    linhas_antes = len(df)
    lista_indices = list(indices_para_remover)
    
    # Remove as linhas problemáticas
    df.dropna(subset=['Class'], inplace=True)
    linhas_depois = len(df)
    
    num_removidas = linhas_antes - linhas_depois
    if num_removidas == 1:
        print(f"Foi removida {num_removidas} linha problemática. Índice da linha: {lista_indices[0]}")
    else:
        print(f"Foram removidas {num_removidas} linhas problemáticas. Índices das linhas: {lista_indices}")
else:
    print("Nenhuma linha com NaN na coluna 'Class' encontrada. Nenhuma linha foi removida.")


# =============================================================================
# 3. ANÁLISE EXPLORATÓRIA RÁPIDA (EDA)
# =============================================================================
print(f"\nO dataset agora possui {df.shape[0]} linhas e {df.shape[1]} colunas.")
print("Verificando a distribuição das classes (0: Legítima, 1: Fraude):")
class_counts = df['Class'].value_counts()
print(class_counts)
fraud_percentage = (class_counts[1] / class_counts.sum()) * 100
print(f"Porcentagem de transações fraudulentas: {fraud_percentage:.4f}%")

# =============================================================================
# 4. PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================
print("\n--- Iniciando o Pré-processamento ---")

df_proc = df.copy()
scaler = StandardScaler()
df_proc['scaled_amount'] = scaler.fit_transform(df_proc['Amount'].values.reshape(-1, 1))
df_proc['scaled_time'] = scaler.fit_transform(df_proc['Time'].values.reshape(-1, 1))
df_proc.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df_proc.drop('Class', axis=1)
y = df_proc['Class']

columns_order = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Dados divididos em conjuntos de treino (80%) e teste (20%).")

# =============================================================================
# 5. TRATAMENTO DO DESBALANCEAMENTO COM SMOTE
# =============================================================================
print("\n--- Tratando o Desbalanceamento de Classes com SMOTE ---")
print(f"Distribuição original no treino: \n{y_train.value_counts()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Distribuição após SMOTE no treino: \n{y_train_resampled.value_counts()}")

# =============================================================================
# 6. TREINAMENTO DO MODELO
# =============================================================================
print("\n--- Treinando o Modelo RandomForestClassifier ---")
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10, n_estimators=100)
rf_model.fit(X_train_resampled, y_train_resampled)
print("Modelo treinado com sucesso.")

# =============================================================================
# 7. PREVISÃO E AVALIAÇÃO
# =============================================================================
print("\n--- Avaliando o Desempenho do Modelo no Conjunto de Teste ---")
y_pred = rf_model.predict(X_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Legítima (0)', 'Fraude (1)']))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)

# =============================================================================
# 8. VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================
print("\n--- Gerando Visualizações ---")

reports_dir = 'reports'
os.makedirs(reports_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legítima Prevista', 'Fraude Prevista'],
            yticklabels=['Legítima Real', 'Fraude Real'])
plt.title('Matriz de Confusão', fontsize=16)
plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, 'matriz_confusao.png'))
print(f"Gráfico 'matriz_confusao.png' salvo na pasta '{reports_dir}'.")

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Features Mais Importantes', fontsize=16)
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, 'importancia_features.png'))
print(f"Gráfico 'importancia_features.png' salvo na pasta '{reports_dir}'.")

# =============================================================================
# 9. SALVANDO O MODELO PARA PRODUÇÃO
# =============================================================================
print("\n--- Salvando o Modelo e o Normalizador ---")

artifacts_dir = 'artifacts'
os.makedirs(artifacts_dir, exist_ok=True)

joblib.dump(rf_model, os.path.join(artifacts_dir, 'fraud_model.pkl'))
print(f"Modelo salvo como 'fraud_model.pkl' na pasta '{artifacts_dir}'")

joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.pkl'))
print(f"Normalizador salvo como 'scaler.pkl' na pasta '{artifacts_dir}'")

columns_path = os.path.join(artifacts_dir, 'columns.json')
with open(columns_path, 'w') as f:
    json.dump(columns_order, f)
print(f"Ordem das colunas salva em '{columns_path}'")

print("\n--- Fim do Script ---")