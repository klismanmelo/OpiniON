import streamlit as st
import torch
from file import *
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd

# Configurar o modelo e tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Carregar o DataFrame
opinios = load_file()
coluna = 0  # Ajustar para a coluna com textos


# Função para análise de sentimentos
def emotion(opinios_user):
    positive, negative = 0, 0
    resultados = []

    # Nome da coluna
    coluna_nome = opinios_user.columns[coluna]
    respostas_usuarios = opinios_user.iloc[:, coluna]

    for resposta in respostas_usuarios:
        # Ignorar valores nulos
        if pd.isna(resposta):
            continue

        # Garantir que é string
        resposta = str(resposta)

        # Tokenizar o texto
        inputs = tokenizer(resposta, return_tensors="pt", truncation=True, padding=True)

        # Fazer predição
        with torch.no_grad():
            logits = model(**inputs).logits

        # Obter o rótulo previsto
        predicted_class_id = logits.argmax().item()
        emotion_label = model.config.id2label[predicted_class_id]

        # Contar emoções
        if emotion_label == "POSITIVE":
            positive += 1
        else:
            negative += 1

        # Adicionar resultado à lista
        resultados.append(f"Texto: {resposta} | Emoção: {emotion_label}")

    return positive, negative, resultados


# Função principal do Streamlit
def main():
    st.title("Sentiment Analysis")
    st.write("Clique no botão abaixo para realizar a análise de sentimentos.")

    # Botão para executar a análise
    if st.button("Predict", use_container_width=True):
        st.write("Analisando sentimentos...")

        # Chamar a função de análise
        positive, negative, resultados = emotion(opinios)

        # Exibir resultados
        st.success(f"Total Positivos: {positive}")
        st.error(f"Total Negativos: {negative}")
        st.write("Detalhes das análises:")
        for resultado in resultados:
            st.write(resultado)


# Executar a aplicação
if __name__ == "__main__":
    main()
