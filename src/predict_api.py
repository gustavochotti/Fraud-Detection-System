# -*- coding: utf-8 -*-
"""
API de Previsão de Fraude com Flask

Este script cria um servidor web simples que expõe o nosso modelo de ML treinado.
Ele possui um endpoint '/predict' que aceita dados de transação em formato JSON,
processa esses dados usando o scaler salvo, faz a previsão com o modelo salvo
e retorna o resultado.

Autor: [Seu Nome Aqui]
Data: 19 de agosto de 2025
"""

import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Inicializa o aplicativo Flask
app = Flask(__name__)

# =============================================================================
# CARREGAR OS MODELOS SALVOS (ARTEFATOS)
# =============================================================================
artifacts_dir = 'artifacts'
model_path = os.path.join(artifacts_dir, 'fraud_model.pkl')
scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Modelo e Scaler carregados com sucesso.")
except FileNotFoundError:
    print(f"Erro: Arquivos não encontrados nos caminhos '{model_path}' ou '{scaler_path}'.")
    print("Execute o script 'train_model.py' primeiro para gerar os artefatos.")
    model = None
    scaler = None

# =============================================================================
# DEFINIR A ORDEM CORRETA DAS COLUNAS
# =============================================================================
COLUMNS_ORDER = [
    'scaled_time', 'scaled_amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
    'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
    'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
]

# =============================================================================
# CRIAR O ENDPOINT DA API
# =============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Recebe dados de transação, processa e retorna a previsão de fraude.
    """
    if not model or not scaler:
        return jsonify({'error': 'Modelo não está carregado. Verifique os logs do servidor.'}), 500

    try:
        json_data = request.get_json()
        
        df_new_transaction = pd.DataFrame(json_data, index=[0])

        df_new_transaction['scaled_amount'] = scaler.transform(df_new_transaction['Amount'].values.reshape(-1, 1))
        df_new_transaction['scaled_time'] = scaler.transform(df_new_transaction['Time'].values.reshape(-1, 1))
        
        df_new_transaction.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        df_processed = df_new_transaction[COLUMNS_ORDER]

        prediction = model.predict(df_processed)
        probability = model.predict_proba(df_processed)

        is_fraud = bool(prediction[0])
        fraud_probability = probability[0][1]
        
        response = {
            'prediction': 'Fraude' if is_fraud else 'Legítima',
            'is_fraud': is_fraud,
            'fraud_probability': f"{fraud_probability:.4f}"
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# =============================================================================
# EXECUTAR O SERVIDOR
# =============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)