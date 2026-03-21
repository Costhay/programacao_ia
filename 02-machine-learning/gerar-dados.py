# Importando as libs necessarias #
import pandas as pd
import numpy as np

# Criando numeros aleatorios #
# Definir uma semente #

np.random.seed(42)

# Gerar 500 registros #

numero_registros = 500

# Estruturar dados do arquivo .csv

data =  {
    'tempo_contrato': np.random.randint(1, 48, numero_registros), #definindo o tempo de 1 a 48 meses e esses dados vao ser popuados em dados registro
    'valor_mensal': np.random.uniform(50.0, 150.0, numero_registros).round(2), # assinaturas que variam de 50 a 150
    'reclamacoes': np.random.poisson(1.5, numero_registros) #cada usuario tem uma media de uma reclamacao e meia
}

# Converter o dicionario de dados em um conjunto de dados

df = pd.DataFrame(data)

# Criar a simulacao da logica de churn
# O cliente tem mais chance de sair de tiver mais reclamacoes, ou se o tempo de contrato for curto

df['cancelou'] = ((df['reclamacoes'] > 2) | (df['tempo_contrato'] < 6)).astype(int)

# Salvar Dataset em .csv

df.to_csv('churn-data.csv', index=False)

print("Arquivo 'churn_data.csv' gerado com sucesso!")