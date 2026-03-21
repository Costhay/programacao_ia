#Importando todos os modulos necessarios
import pandas as pd 
#criar e alterar dados em tabelas
import numpy as np 
#ferramenta de analise matematica

from sklearn.preprocessing import StandardScaler 
#organizar os numeros para que numeros muito pequenos e muito grandes fiquem na mesma escala
from sklearn.ensemble import RandomForestClassifier 
#para que a IA consiga analisar as churns, trabalha como uma arvore agrupando ate chegar em um
from sklearn.metrics import classification_report, confusion_matrix 
#vao criar um relatorio de classificacao para conseguir medir se a IA e eficiente ou nao, e gerar um grafico para ver se esta acertivo, atraves dos dados de teste e de treino
from sklearn.model_selection import train_test_split 
#modulo responsavel por permitir que o modelo seja dividido, entre teste e treino

import seaborn as sns
import matplotlib.pyplot as pyplot
import joblib

#etapa 2: importar o dicionario com dados

try:
    print("Carregando arquivo 'churn-data.csv'...")
    df = pd.read_csv('churn-data.csv') 
    # ler o arquivo e criar uma tabela
    print(f"Sucesso, {len(df)} linhas importadas")

except FileNotFoundError:
    print("O arquivo não pode ser encontrado na pasta")
    exit()

# etapa 3: pre processamento de dados (preparar a ia pra ser treinada)
# passo 1: separar a pergunta (x) da resposta (Y)
# (X) é tudo menos a coluna cancelou, sao as "pistas" pro modelo

X = df.drop('cancelou',axis=1) 
#considerar todas as coluna menos as que estiverem em argumento

# (Y) sao as respostas, apenas a coluna cancelou, o que queremos que o modelo preveja

y = df['cancelou']

#passo 2: dividir o treino do teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#uma quantidade de linhas para teste, e de colunas tambem. Elas serao divididas pelo traintestsplit, de acordo com a logica definida dentro dos argumentos, nesse caso 20% dos dados, random state é uma variavel para que os dados sejam embaralhados 42 vezes, para que seja validado de maneira adequada

# Passo 3: normalizando (colocando na mesma escala)

scaler = StandardScaler()

# fit transform do treino: IA calcula media e desvpad dos dados de treino - zip scaler - padronizacao

X_train_scaled = scaler.fit_transform(X_train)

#fit transform no teste: usamos a regua calculada no treino

X_test_scaled = scaler.transform(X_test)

# etapa 4: treinar o modelo e realizar a previsao de dados - CORE DO MODELO
# criando o modelo

modelo_churn = RandomForestClassifier(n_estimators=100, random_state=42) 
# criando 100 arvores de decisoes - embaralha os dados de forma aleatoria 42 vezes

# treinar e ajustar a IA - apender os padros

modelo_churn.fit(X_train_scaled, y_train)

# prever as respostas

previsoes = modelo_churn.predict(X_test_scaled)

#Etapa 5: avaliacao do modelo

print("Relatório de Performance") 
# comparar o gabarito com o que a IA previu

print(classification_report(y_test, previsoes))

# Etapa 6: Deploy -  salvar o trabalho
joblib.dump(modelo_churn, 'modelo.churn_v1.pkl')
#extensao do arquivo pkl
#primeiro chama a variavel e depois nomeia como ela vai se chamar
#Precisa salvar a regua tambem
joblib.dump(scaler, 'padronizador_v1.pkl')
#vai gerar dois arquivos, e sao obrigatorios para trabalhar com modelo de machine leraning
# esses arquivos serao os importados
print("Arquivos de ML foram exportados com sucesso")