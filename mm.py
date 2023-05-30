from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
from langdetect import detect
from translate import Translator

translator = Translator(to_lang="fr")

# Créer une colonne pour l'image
# Définir le style CSS pour déplacer l'image vers la gauche

st.subheader("Chatbot - Édition marocaine rapport 2022")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Quelle information souhaitez-vous obtenir du rapport ?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["open_api_key"])

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()
query = st.text_input("Posez votre question au chatbot :")
with textcontainer:
    logo_path = "medias24.png"

# Afficher l'image dans la barre latérale
    st.sidebar.image(logo_path, width=200) 
    st.sidebar.subheader("Suggestions:")
    st.sidebar.markdown("##### choisir:")
    questions = [
    "L'idée générale du rapport ?", 
    "Quels sont les principaux chiffres et tendances de l'édition marocaine en 2022 ?",
    "Comment se fait la collecte et l'aquisition des publication marocaine ?",
    "Quel est la répartition de la publications des livres selon les langues ?",
    "Comment se répartit géographiquement l'activité éditoriale au maroc ?"
    
]

    selected_questions = []

    for question in questions:
     if st.sidebar.checkbox(question):
        selected_questions.append(question)  
    if selected_questions:
        for selected_question in selected_questions:
          question = selected_question
        if question:
         with st.spinner("En train de taper..."):
            context = find_match(question)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{question}")
            if detect(response)=='en':
                response = translator.translate(response)
         st.session_state.requests.append(question)
         st.session_state.responses.append(response)
    elif query:
        with st.spinner("En train de taper..."):

            context = find_match(query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            if detect(response)=='en':
                response = translator.translate(response)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
        
 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
