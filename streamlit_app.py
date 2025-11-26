import streamlit as st
from src.retriever import Retriever
from src.rag import generate_with_hf, generate_with_openai

st.title('RAG QA Demo')
st.write('Simple Retrieval-Augmented Generation demo for thesis.')

index_path = st.text_input('FAISS index path', 'models/faiss.index')
meta_path = st.text_input('FAISS meta path', 'models/faiss.index.meta.pkl')
model_choice = st.selectbox('Generator backend', ['hf','openai'])
query = st.text_input('Question', '')

if st.button('Run') and query.strip():
    retriever = Retriever(index_path, meta_path)
    contexts = retriever.retrieve(query, k=5)
    st.subheader('Retrieved')
    for c in contexts:
        st.markdown(f"**Score:** {c['score']:.4f} — *{c.get('source')}*\n\n{c['text']}")
    st.subheader('Answer')
    if model_choice == 'openai':
        ans = generate_with_openai(query, contexts)
    else:
        ans = generate_with_hf(query, contexts)
    st.write(ans)
