import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Use langchain_openai imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# Import Document class for type hinting & creation if needed later
from langchain.docstore.document import Document
# Note: Ensure Graphviz system software is installed if using diagram feature


# --- Utility Functions ---

@st.cache_data(show_spinner=False)
def clone_repository(repo_url: str, repo_dir: str) -> None:
    """Clone the repository if the target directory does not exist."""
    if not os.path.exists(repo_dir):
        st.write(f"Cloning repository {repo_url}...")
        # Consider adding error handling for git clone command
        os.system(f"git clone --depth 1 {repo_url} {repo_dir}") # Use --depth 1 for faster clone
        st.write("Repository cloned.")
    else:
        st.write(f"Repository directory '{repo_dir}' already exists.")
    return

@st.cache_data(show_spinner=False)
def load_repository_docs(repo_dir: str):
    """Load documents from the repository using a DirectoryLoader."""
    st.write(f"Loading documents from repository: {repo_dir}")
    if not os.path.exists(repo_dir):
        st.error(f"Repository directory not found: {repo_dir}. Please ensure it's cloned.")
        return [] # Return empty list if dir doesn't exist

    loader = DirectoryLoader(
        repo_dir,
        glob=["**/*.py", "**/*.rst", "**/*.yaml", "**/*.go", "**/Dockerfile"],
        show_progress=True,
        use_multithreading=True, # Optional: keep if helpful
        silent_errors=False # Optional: suppress errors for unreadable files
    )
    try:
        docs = loader.load()
        st.write(f"Loaded {len(docs)} documents from repository.")
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        docs = [] # Return empty list on error
    return docs

@st.cache_resource(show_spinner=False)
def init_qa_chain(repo_dir: str):
    """Initialize the conversational retrieval chain using LangChain."""
    # 1. Load documents from the repository (uses @st.cache_data)
    repo_docs = load_repository_docs(repo_dir)

    # --- Placeholder: Add description docs if using ---
    # desc_docs = create_description_docs() if MODEL_DESCRIPTIONS else []
    # all_docs = repo_docs + desc_docs
    all_docs = repo_docs # Using only repo docs for now
    # --- End Placeholder ---

    if not all_docs:
        st.error("No documents loaded. Cannot initialize QA chain.")
        return None # Return None if no documents

    st.write(f"Total documents to process: {len(all_docs)}")

    # 2. Split documents into chunks
    st.write("Splitting documents into chunks...")
    # Adjusted chunk size based on previous code version
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_docs)
    st.write(f"Split into {len(texts)} text chunks.")

    if not texts:
        st.error("No text chunks generated after splitting. Cannot initialize QA chain.")
        return None # Return None if no chunks

    # 3. Create embeddings and FAISS vector store
    st.write("Creating embeddings and vector store (this may take a moment)...")
    try:
        # Ensure OPENAI_API_KEY is set in environment or Streamlit secrets
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        st.write("Vector store created successfully.")
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        st.error("Please ensure your OpenAI API key is configured correctly.")
        return None # Return None on error

    # 4. Initialize LLM and QA chain
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o") # Ensure API key is available
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vectorstore.as_retriever(),
            return_source_documents=True
        )
        st.write("QA chain initialized.")
        return qa_chain
    except Exception as e:
        st.error(f"Failed to initialize LLM or QA chain: {e}")
        st.error("Please ensure your OpenAI API key is valid and you have access to the model.")
        return None # Return None on error

# --- Streamlit App Layout ---

st.markdown("<h2 style='text-align: center;'>Pilot: GCIMS AI Foresight Plugin</h2>", unsafe_allow_html=True)
# st.markdown("""
# Ask questions about the `scalable` repository's code and documentation.

# **Tip:** When asking for code related to the `scalable` repository, be specific about the task or functions. The chatbot will try to use examples from the repository's content in its response (e.g., 'Generate Python code to set up the configuration using scalable examples.'). You can also ask it to generate diagrams (e.g., 'Show dependencies as a Graphviz diagram.').
# """)


# Repository settings
REPO_URL = "https://github.com/JGCRI/scalable"
REPO_DIR = "scalable"  # local folder where the repo will be cloned

# Clone repository if needed
clone_repository(REPO_URL, REPO_DIR)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Initialize placeholder for latest source documents (Fixes AttributeError)
if "latest_source_docs" not in st.session_state:
    st.session_state.latest_source_docs = []
# Initialize qa_chain if needed
if "qa_chain" not in st.session_state:
    with st.spinner("Initializing chatbot knowledge base..."):
        # Initialize chain only if directory exists
        if os.path.exists(REPO_DIR):
             st.session_state.qa_chain = init_qa_chain(REPO_DIR)
        else:
             st.session_state.qa_chain = None # Set to None if repo dir missing

    # Check if chain initialization was successful
    if st.session_state.qa_chain:
        st.success("Chatbot initialized successfully!")
    else:
        st.error("Chatbot initialization failed. Please check logs and configuration.")
# --- End Session State Initialization ---


# User input - Assign a key to allow clearing it later if desired
query = st.text_area("Enter your question:", height=100, key="query_input", disabled=(st.session_state.qa_chain is None))

# Main interaction logic
if st.button("Send", disabled=(st.session_state.qa_chain is None)) and query:
    with st.spinner("Thinking..."):
        try:
            input_data = {
                "question": query,
                "chat_history": st.session_state.chat_history
            }
            result = st.session_state.qa_chain.invoke(input_data)

            answer = result.get("answer", "Sorry, I couldn't find an answer.")
            # Store latest source docs in session state
            st.session_state.latest_source_docs = result.get("source_documents", [])
        except Exception as e:
            st.error(f"Error during QA chain processing: {e}")
            answer = "Sorry, an error occurred while processing your request."
            st.session_state.latest_source_docs = [] # Clear sources on error

    # Update chat history for context (even if an error occurred)
    st.session_state.chat_history.append((query, answer))

    # Optional: Clear the input box after sending. Requires rerun.
    # st.session_state.query_input = ""
    # st.rerun()


# --- Display Section ---
st.markdown("### Conversation")

# Check if chat history exists and is not empty
if st.session_state.chat_history:
    # Iterate through chat history indices in reverse order (newest first)
    for i in range(len(st.session_state.chat_history) - 1, -1, -1):
        q, a = st.session_state.chat_history[i]

        # --- Display Question and Answer ---
        st.markdown(f"**You:** {q}")

        is_latest_message = (i == len(st.session_state.chat_history) - 1)
        # Ensure 'a' is a string before checking startswith
        is_diagram_code = isinstance(a, str) and (a.strip().casefold().startswith("graph"))

        # Attempt to render diagram ONLY for the latest message if it looks like DOT code
        if is_latest_message and is_diagram_code:
            st.markdown("**Chatbot:** (Generated Diagram)")
            try:
                st.graphviz_chart(a)
            except Exception as e:
                st.error(f"Failed to render diagram: {e}")
                st.warning("Displaying raw diagram code instead:")
                st.code(a, language='dot') # Show code on error
        else:
             # Display normal text answer (could be old DOT code as text)
             st.markdown(f"**Chatbot:** {a}")
        # --- End Display Question and Answer ---


        # --- Display sources ONLY for the most recent message ---
        if is_latest_message:
            source_docs = st.session_state.latest_source_docs # Use stored sources
            if source_docs:
                st.markdown("---") # Separator before sources section
                st.markdown("##### Sources used for the latest answer:")
                try:
                    # Handle potential variations in metadata structure
                    unique_sources = set()
                    for doc in source_docs:
                        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                           unique_sources.add(doc.metadata.get("source", "Unknown Source"))
                        else:
                           unique_sources.add("Source metadata unavailable")

                    for source in sorted(list(unique_sources)):
                        st.markdown(f"- `{source}`")

                    # Expander for snippets
                    with st.expander("Show Source Snippets", expanded=False):
                        for snip_idx, doc in enumerate(source_docs):
                            meta = getattr(doc, 'metadata', {}) # Safely get metadata
                            source_file = meta.get('source', 'Unknown Source') if isinstance(meta, dict) else 'Unknown Source'
                            content = getattr(doc, 'page_content', '') # Safely get content
                            content_snippet = content[:200] + "..." if len(content) > 200 else content
                            st.markdown(f"**Snippet {snip_idx+1} ({source_file}):**")
                            st.markdown(f"```\n{content_snippet}\n```")
                            # Add separator between snippets, but not after the last one
                            if snip_idx < len(source_docs) - 1:
                                st.markdown("---")
                except Exception as e:
                    st.warning(f"Could not display source documents: {e}")
        # --- End Display sources ---

        # Separator between conversation turns
        st.markdown("---")

# --- End Display Section ---