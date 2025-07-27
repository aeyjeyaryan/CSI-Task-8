import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load your secret Google API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# UI: Sidebar
st.set_page_config("💡 ISHA Learning", layout="wide", page_icon=":sparkles:")
st.sidebar.title("Options")
st.sidebar.info("Upload your PDFs or use predefined python tutorial ones!")

# Predefined PDFs for demo (update paths to yours)
PREDEFINED_PDFS = ["a.pdf","b.pdf","c.pdf","d.pdf","e.pdf"]

# --- File management ---
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

@st.cache_data(show_spinner=False)
def read_pdfs(pdf_files):
    all_text = ""
    for file in pdf_files:  # Accepts both uploaded file-like objects or paths
        if isinstance(file, str):
            file = open(file, "rb")
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
        if not isinstance(file, str):
            file.close()
    return all_text

@st.cache_data(show_spinner=False)
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    return splitter.split_text(text)

@st.cache_resource(show_spinner=False)
def build_vector_db(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embedding=embeddings)
    return db

def create_prompt():
    return PromptTemplate(
        template="""You are a helpful assistant. Answer as detailed as possible using the provided context. 
If the answer is not in context, reply: "Answer is not available in the context".
\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:""",
        input_variables=["context", "question"]
    )

def get_chain(vector_db):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.25)
    prompt = create_prompt()
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain

with st.spinner("Preparing knowledge base..."):
    pdf_source = uploaded_files if uploaded_files else PREDEFINED_PDFS
    raw_text = read_pdfs(pdf_source)
    text_chunks = split_text(raw_text)
    vector_db = build_vector_db(text_chunks)
    chain = get_chain(vector_db)

st.title("Document RAG Model")

# UI: Question input
col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_input("Ask any question about the documents...", placeholder="Enter your question here")
with col2:
    st.markdown(" ")

if user_query:
    with st.spinner("Thinking..."):
        response = chain({"query": user_query})
        st.write("### Answer:")
        st.success(response["result"])

        # Display supporting docs
        with st.expander("Show supporting PDF context"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Chunk {i+1}:** {doc.page_content[:800]}{'...' if len(doc.page_content) > 800 else ''}")


