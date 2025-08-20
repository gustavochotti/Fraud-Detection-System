# -*- coding: utf-8 -*-
"""
Script de Predição em Lote a partir de um Arquivo CSV

Este script carrega o modelo de detecção de fraude treinado e o utiliza para
fazer previsões em um lote de novas transações fornecidas em um arquivo CSV.

Como usar:
No terminal, na pasta raiz do projeto, execute:
python src/predict_from_csv.py --filepath data/novas_transacoes.csv

Autor: [Seu Nome Aqui]
Data: 19 de agosto de 2025
"""

import os
import argparse
import json
import joblib
import pandas as pd

def predict_batch(filepath):
    """
    Carrega o modelo, processa os dados do CSV e imprime as previsões.
    """
    # =========================================================================
    # CARREGAR OS ARTEFATOS
    # =========================================================================
    artifacts_dir = 'artifacts'
    try:
        model = joblib.load(os.path.join(artifacts_dir, 'fraud_model.pkl'))
        scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
        with open(os.path.join(artifacts_dir, 'columns.json'), 'r') as f:
            columns_order = json.load(f)
        print("Modelo, Scaler e Ordem das Colunas carregados com sucesso.")
    except FileNotFoundError:
        print(f"Erro: Artefatos não encontrados na pasta '{artifacts_dir}'.")
        print("Execute o script 'src/train_model.py' primeiro.")
        return

    # =========================================================================
    # CARREGAR E VALIDAR O NOVO DATASET
    # =========================================================================
    try:
        new_data = pd.read_csv(filepath)
        print(f"Arquivo '{filepath}' carregado. {len(new_data)} transações para analisar.")
    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada não encontrado em '{filepath}'.")
        return
        
    # Salva uma cópia dos dados originais para a exibição final
    original_data = new_data.copy()

    # =========================================================================
    # PRÉ-PROCESSAMENTO DOS NOVOS DADOS
    # =========================================================================
    print("Iniciando pré-processamento dos novos dados...")
    # Verifica se as colunas necessárias existem
    required_raw_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    if not all(col in new_data.columns for col in required_raw_cols):
        print("Erro: O arquivo CSV de entrada não contém todas as colunas necessárias.")
        print(f"Colunas esperadas: {required_raw_cols}")
        return

    # Aplica a mesma transformação de normalização
    new_data['scaled_amount'] = scaler.transform(new_data['Amount'].values.reshape(-1, 1))
    new_data['scaled_time'] = scaler.transform(new_data['Time'].values.reshape(-1, 1))
    
    new_data.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Garante a ordem correta das colunas
    df_processed = new_data[columns_order]
    
    # =========================================================================
    # FAZER AS PREVISÕES
    # =========================================================================
    print("Fazendo previsões...")
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)
    
    # Pega a probabilidade de ser fraude (classe 1)
    fraud_probabilities = probabilities[:, 1]
    
    # =========================================================================
    # EXIBIR OS RESULTADOS
    # =========================================================================
    print("\n--- Resultados da Análise de Fraude ---")
    original_data['Prediction'] = ['Fraude' if p == 1 else 'Legítima' for p in predictions]
    original_data['Fraud_Probability'] = [f"{prob:.2%}" for prob in fraud_probabilities]
    
    # Imprime o resultado no terminal
    # .to_string() garante que todas as colunas e linhas sejam exibidas
    print(original_data[['Time', 'Amount', 'Prediction', 'Fraud_Probability']].to_string())


if __name__ == '__main__':
    # Configura o parser para aceitar argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Analisa um CSV de transações para detectar fraudes.")
    parser.add_argument('-f', '--filepath', type=str, required=True, help="Caminho para o arquivo CSV com as novas transações.")
    
    args = parser.parse_args()
    
    predict_batch(args.filepath)