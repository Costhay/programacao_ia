#Import das bibliotecas necessárias
import pandas as pd     #Para manipulação de dados na forma de tabela
import spacy        #Lib de processamento de linguagem natural
import joblib       #salvar e carregar modelos de IA treinados
from sklearn.feature_extraction.text import TfidfVectorizer     #Converte texto em vetores de palavras
from sklearn.naive_bayes import MultinomialNB       #Classifica o texto em categorias com base em probabilidade -> muito utilizada nas ferramentas de detecção de spam
from sklearn.pipeline import make_pipeline      #Junta várias etapas em um único fluxo
from sklearn.model_selection import train_test_split        #Divide o conjunto de dados em treino e teste
from sklearn.metrics import classification_report       #Avalia o modelo

#Etapa 1: Carregar os dados
print("Carregando o dataset...")
df = pd.read_csv("dataset_chamados.csv")

#Etapa 2: Pipeline de processamento focaco em performance
#Usaremos o Spacy dentro do fluxo da UI

nlp = spacy.load("pt_core_news_sm") #Carregamento da SpaCy em ptbr

def prep(texto):
    doc = nlp(texto) #processamento do texto (tokenização e análise probabilística)

    return " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_punct   #remove todo tipo de pontuação
    ])

print("O processamento do texto pode levar alguns instantes.")

df['texto_limpo'] = df['texto'].apply(prep)   #Aplicar a função de limpeza na coluna de texto

#Etapa 3: Dividir entre treino e teste
#X = textos de entrada
#y = categorias (labels)
X_train, X_test, y_train, y_test = train_test_split(
    df['texto_limpo'],      #Dados de entrada pré-processados
    df['categoria'],     #Categorias
    test_size = 0.2     #20% para teste
)

#Etapa 4: criar e treiar pipeline de ML
model_pipeline = make_pipeline(
    TfidfVectorizer(),      #converter texto em valores numéricos
    MultinomialNB()     #aplica classificador Naive Bayes (palavra: intenção/categoria)
)

#Treinar modelo com os dados de treino
model_pipeline.fit(X_train, y_train)

#Etapa 5: Salvar modelo treinado
joblib.dump(model_pipeline, "modelo_triagem_suporte.pkl")
print("Modelo treinado e salvo como modelo_triagem_suporte.pkl")

