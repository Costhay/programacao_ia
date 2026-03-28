import streamlit as st
import joblib
import spacy
import pandas as pd

#configuração da pagina (titulo)
st.set_page_config(page_title="Triagem de chamados", page_icon="J")

#carregamento de recursos - em cache para que não seja necessário carregar a cada clique

@st.cache_resource
#funcao que vai retornar sempre que necessario o modelo criado

def carregar_modelo():
    return joblib.load("modelo_triagem_suporte.pkl") #carregamento do modelo ml treinado

@st.cache_resource

def carregar_nlp():
    return spacy.load("pt_core_news_sm")

try:
    modelo = carregar_modelo()
    nlp = carregar_nlp()

except:
    st.error("Erro: Execute o script 'treinar_modelo_py' para gerar o arquivo .pkl")
    st.stop()

# logica de processamento

def analisar_chamado(texto_usuario):

    #1-processamento linguistico com spacy
    doc = nlp(texto_usuario)

    #extracao de entidades nomeadas (palavras importantes, com algum sentido)
    entidades = [(ent.text,ent.label_) for ent in doc.ents]
    
    #limpeza de texto para o modelo em 3 etapas: lematizar, converter minusculo, remover pontuação
    texto_limpo = " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_punct
    ])

    #2-predicao com o modelo de ML
    #prever a categoria do chamado
    categoria_predita = modelo.predict([texto_limpo])[0] #consiedera o primeiro elemento do array

    #probabilidade de cada categoria (classe de chamado)
    probs = modelo.predict_proba([texto_limpo])[0]

    #pegar a que tem a maior probabilidade de acontecer como resultado da analise
    confianca = max(probs)*100 
    #retorna os resultados
    return categoria_predita, confianca, entidades

# interface gratica -----------
st.title("Triagem de suporte")
st.markdown("Descreva o problema em poucas palavras")

#criando o chat - se nao tem mensagem, nao tem mensagem, ele é retroalimentado
if "messages" not in st.session_state:
    st.session_state.messages = []

#exibir mensagens anteriores no chat
#reconstroi todo o historico de mensagens existentes
#caracteristica do streamlit recarregar todo o codigo a cada nova interaçao, então com isso criamos uma memoria temporaria dentro do streamlit para cada mensagem
#percorre cada mensagem, carrega o historico de mensagem e exibe dentro do balao de mensagem
for message in st.session_state.messages:
    with st.chat_message(message["role"]): #criando um botao de chat visual
        st.markdown(message["content"]) #exbindo o texto da mensagem dentro do balao

#exibir o campo de entrada para o usuario (chat input)
if prompt:= st.chat_input("Ex.: O servidor AWS parou de responder..."):

    st.chat_message('user').markdown(prompt)

    st.session_state.messages.append({
        "role":"user",
        "content": prompt
    })

#processar a resposta que a IA (modelo ML) vai retornar ao usuario
#prompt a mensagem que o usuario escreveu no input

    categoria, confianca, ents = analisar_chamado(prompt)

#montar/personalizar a resposta em um formato amigavel (markdown)
    resposta_md = f"""
    **Analise do chamado:**
    **Categoria:** `{categoria}`
    **Confianca:** `{confianca:.2f}%`
    """

    if ents:
        resposta_md += "\n\n **Entidades detectadas:**"
        for ent in ents:
            resposta_md += f"\n-*{[0]}*({ent[1]})"

#ações automaticas por categoria
    acoes = {
        "infraestrutura": "Encaminhando para equipe N2",
        "Acesso": "Verificando logs de autenticação",
        "Hardware": "Abrindo ordem de serviço",
        "Software": "Verificando disponibilidade de licenças"
    }
    #Adicionar ações sugeridas com base na categoria
    resposta_md += f"\n\n **Ação:** {acoes.get(categoria, 'Triagem manual necessária')}"

#3-exibir a resposta do assistente
    with st.chat_message('assistant'):
        st.markdown(resposta_md)

    #salvar resposta no historico, guarda dentro de um vetor
    st.session_state.messages.append({
        "role":"assistant",
        "content": resposta_md
    })
