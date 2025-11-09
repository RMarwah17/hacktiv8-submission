import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Fungsi untuk load LLM (Mistral-7B quantized)
@st.cache_resource
def load_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True  # Quantization untuk hemat RAM
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1
    )
    return HuggingFacePipeline(pipeline=pipe)

# Fungsi untuk proses PDF dan buat vector store
def process_pdf(uploaded_file):
    loader = PyPDFLoader(uploaded_file)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    return db

# UI Streamlit
st.title("📄 PDF Q&A Chatbot")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        db = process_pdf(uploaded_file)
        llm = load_llm()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever()
        )
        st.success("PDF processed! Ask questions below.")
    
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain.run(query)
            st.write("**Answer:**")
            st.write(result)
