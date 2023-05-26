from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
openai.api_key = st.secrets["open_api_key"]
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key=st.secrets["pincone_key"], environment=st.secrets["environement"])
index = pinecone.Index('testchat')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

