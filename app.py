import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# --- Utility Functions ---

@st.cache_data(show_spinner=False)
def clone_repository(repo_url: str, repo_dir: str) -> None:
    """Clone the repository if the target directory does not exist."""
    if not os.path.exists(repo_dir):
        st.write("Cloning repository...")
        os.system(f"git clone {repo_url} {repo_dir}")
    return

@st.cache_data(show_spinner=False)
def load_repository_docs(repo_dir: str):
    """Load documents from the repository using a DirectoryLoader.
    
    Adjust the glob pattern if you wish to restrict the loaded files.
    Here we load all files (excluding hidden directories like .git).
    """
    loader = DirectoryLoader(
        repo_dir, 
        glob=["**/*.py", "**/*.rst", "**/*.yaml", "**/*.go", "**/Dockerfile"],
        show_progress=True
    )
    docs = loader.load()
    return docs

@st.cache_data(show_spinner=False)
def get_vectorstore(docs):
    """Split documents into chunks and create a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()  # Ensure OPENAI_API_KEY is set in your environment
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

@st.cache_resource(show_spinner=False)
def init_qa_chain(repo_dir: str):
    """Initialize the conversational retrieval chain using LangChain."""
    docs = load_repository_docs(repo_dir)
    vectorstore = get_vectorstore(docs)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        vectorstore.as_retriever(), 
        return_source_documents=True
    )
    return qa_chain

# --- Streamlit App Layout ---

st.title("Repository Chatbot")
st.markdown("This chatbot lets you ask questions about the repository's contents.")

# Repository settings
REPO_URL = "https://github.com/JGCRI/scalable"
REPO_DIR = "scalable"  # local folder where the repo will be cloned

# Clone repository if needed
clone_repository(REPO_URL, REPO_DIR)

# Initialize conversation state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    with st.spinner("Setting up the Q&A chain..."):
        st.session_state.qa_chain = init_qa_chain(REPO_DIR)

# User input
query = st.text_input("Enter your question about the repository:")

if st.button("Send") and query:
    with st.spinner("Processing..."):
        result = st.session_state.qa_chain({
            "question": query, 
            "chat_history": st.session_state.chat_history
        })
    answer = result.get("answer", "No answer returned.")
    # Update chat history for context
    st.session_state.chat_history.append((query, answer))
    
    # Display the conversation
    st.markdown("### Conversation")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
    
    # Optionally display source documents for transparency
    source_docs = result.get("source_documents", [])
    if source_docs:
        st.markdown("#### Source Documents")
        for doc in source_docs:
            meta = doc.metadata
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            st.markdown(f"**{meta}**: {content}")
