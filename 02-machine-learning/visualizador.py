#importar modulos

import streamlit as st
import joblib 
import numpy as np

#passo 1 - configurando a aba do navegador
st.set_page_config(page_title = "Analise Churn", page_icon="🥸")

#textos da tela principal
st.title("Sistema de retenção de base")
st.markdown("Insira dados do cliente para verificar risco de cancelamento")

#Passo 2: Importar os passos da IA com joblib (ele cria um arquivo .pkl gerado no script anterior modelo-ml.py)

modelo = joblib.load('modelo.churn_v1.pkl')
# se estivesse em outra pasta, precisaria pegar o caminho do arquivo, e precisa digitar com a extensao. Carrega as regras de decisão do modelo

#Implementar o padronizador para criar uma regua matematica do modelo. 
scaler = joblib.load('padronizador_v1.pkl')

#Passo 3: criar formulario de entrada - interface

col1, col2 = st.columns(2) #criando duas colunas

#coluna esquerda - col1
with col1:
    tempo = st.number_input("Tempo de contrato (meses)", min_value=1, value=12, max_value=200)
    valor = st.number_input("Valor da assinatura: (R$)", min_value=0.0, value=50.0)

with col2:
    reclamacoes = st.slider("Historico de reclamações", 0,10,1) #escala

#Passo 4: realizar o processamento de dados
if st.button("Analisar ricso"):
    dados = scaler.transform([[tempo, valor, reclamacoes]]) 
    #Fazer ajuste para que tenham o mesmo peso, padrao
    probabilidade = modelo.predict_proba(dados)[0][1] 

#Passo 5: feedback de negocio - relacionado a saida que a propria IA gera
    st.divider() #cria uma linha divisoria

#probabilidade de 70%
    if probabilidade > 0.7:
        st.error(f"*ALTO RISTO DE CHURN* ({probabilidade*100:.1f}%)")
        st.info ("*Sugestão de ação:* Oferecer cupom de fidelidade FID210360OFF") #traz uma sugestao

    elif probabilidade > 0.3:
        st.warning(f"*Risco moderado de churn* ({probabilidade*100:.1f}%)")
        st.info("*Sugestão de ação: *Realizar chamada de acompanhamento.")

    else:
        st.success(f"*Cliente estável dentro da plataforma* ({probabilidade*100:.1f}%)")







# estou com sono