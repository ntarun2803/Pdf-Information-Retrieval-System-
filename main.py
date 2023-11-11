import langchain.memory
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os

os.environ['GOOGLE_API_KEY'] = ' '


def extract_pdf_text(pdf_docs):
    text = ""
    for doc in pdf_docs:
        pdfreader = PdfReader(doc)
        for page in pdfreader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = langchain.memory.ConversationBufferMemory(memory_key = "chat_history",return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)
    return conversation_chain

def user_input(question):
    response = st.session_state.conversation({'question' : question})
    st.session_state.chatHistory = response['chat_history']
    for i,message in enumerate(st.session_state.chatHistory):
        st.write(i,message)


def main():
    st.set_page_config("chat with multiple pdfs")
    st.header("Multiple Pdf Information Retreival System (MPIRS)")
    uques = st.text_input("Ask a Question")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    if uques:
        user_input(uques)



    with st.sidebar:
        st.title("Configurations")
        st.subheader("Upload Documents Here")
        pdf_docs = st.file_uploader("Upload Files",accept_multiple_files=True)
        if st.button("Proceed"):
            with st.spinner("Processing"):
                raw_text = extract_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("done")







if __name__ == "__main__":
    main()
