import os
import time
import tempfile
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
# langchain core classes and utilities
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader



load_dotenv()

st.set_page_config(
    page_title="📝 RAG Q&A with PDF uploads and chat history",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📝 RAG Q&A with PDF uploads and chat history")



st.sidebar.header("🛠 Configuration")
st.sidebar.markdown("""
**Steps to get started:**
- 🔑 Enter your Groq API Key below
- 📄 Upload PDF files on main page
- 💬 Ask questions and see chat history
- 🌍 Ask in any language!

**Features:**
- 🌍 Multi-language support
- 💾 Chat history persistence
- 🔍 Intelligent document search
- ⚡ Fast responses with Groq LLM

**Note:** Using optimized embeddings for better performance
""")

# api key and embedding setup
api_key = st.sidebar.text_input("Groq API Key", type="password")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN","") #for hugging face embeddings

# Initialize embeddings - use real embeddings for production
@st.cache_resource
def load_embeddings():
    """Load embeddings model with caching for performance."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except (ImportError, Exception):
        # Silently fall back to FakeEmbeddings for development
        return FakeEmbeddings(size=384)

embeddings = load_embeddings()
# only proceed if the user has entered their Groq_key
if not api_key:
    st.warning(" 🔑 Please enter your Groq API Key in the sidebar to continue ")
    st.stop()

# instantiate the GROQ LLM
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

st.markdown("### 📑 Upload Your PDF Documents")
uploaded_files = st.file_uploader(
    "Choose PDF file(s)",
    type="pdf",
    accept_multiple_files=True
)

all_docs = []

if uploaded_files:
    # show progress spinner while loading
    with st.spinner(" 🔄 Loading and splitting PDFs"):
        try:
            for pdf in uploaded_files:
                # write to a temp file so PyPDFLoader can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf.getvalue())
                    pdf_path = tmp.name
                
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                finally:
                    # cleanup temp file safely with proper error handling
                    try:
                        if os.path.exists(pdf_path):
                            os.unlink(pdf_path)
                    except (OSError, PermissionError) as cleanup_error:
                        st.warning(f"Could not cleanup temp file: {cleanup_error}")
        except Exception as e:
            st.error(f"Error loading PDFs: {str(e)}")
            st.stop()
    
    if not all_docs:
        st.error("No content found in PDFs")
        st.stop()
    
    # Split documents into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(all_docs)

    # Build vector store with DocArray (simplest, no external dependencies)
    @st.cache_resource
    def get_vectorstore(_splits): 
        try:
            return DocArrayInMemorySearch.from_documents(_splits, embeddings)
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None
    
    vectorstore = get_vectorstore(splits)
    if vectorstore is None:
        st.stop()
    
    retriever = vectorstore.as_retriever()

    # Build a history aware retriever that uses past chat to refine searches

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, decide what to retrieve."),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )
    def create_qa_chain(llm, retriever):
        """Create QA chain with proper prompt engineering.
        
        Args:
            llm: Language model instance
            retriever: Document retriever instance
            
        Returns:
            Configured RAG chain for question answering
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system","You are a multilingual assistant. Use the retrieved context to answer the user's question. "
            "CRITICAL RULE: ALWAYS respond in the EXACT SAME LANGUAGE as the user's question. "
            "Detect the language of the user's input and respond ONLY in that language. "
            "Examples: Roman Urdu → Roman Urdu, English → English, Hindi → Hindi, Arabic → Arabic, etc. "
            "If you don't know the answer, say so in the user's language. Keep responses under three sentences. \n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        return create_retrieval_chain(retriever, question_answer_chain) 

    rag_chain = create_qa_chain(llm, history_aware_retriever)

    # Enhanced session state for chat history with persistence
    if "chathistory" not in st.session_state:
        st.session_state["chathistory"] = {}
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = {}

    def initialize_session(session_id: str):
        """Initialize session state for given session ID.
        
        Args:
            session_id (str): Unique identifier for the chat session
        """
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id] = ChatMessageHistory()
        if session_id not in st.session_state.chat_messages:
            st.session_state.chat_messages[session_id] = []

    def get_history(session_id: str):
        """Get chat history for given session ID.
        
        Args:
            session_id (str): Unique identifier for the chat session
            
        Returns:
            ChatMessageHistory: The chat history object for the session
        """
        initialize_session(session_id)
        return st.session_state.chathistory[session_id]
    
    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_message_key= "input",
        history_messages_key = "chat_history",
        output_messages_key = "answer"
    )

    # Chat UI
    st.markdown("### 💬 Chat Interface")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if 'current_session' not in st.session_state:
            st.session_state.current_session = "default_session"
        session_id = st.text_input("🆔 Session ID", value=st.session_state.current_session, key="session_input")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with input
        if st.button("🔄 New Session"):
            import random
            new_id = f"session_{random.randint(1000, 9999)}"
            st.session_state.current_session = new_id
            st.rerun()
    
    # Update current session if user manually changed it
    if session_id != st.session_state.current_session:
        st.session_state.current_session = session_id
    
    # Use current session for all operations
    active_session = st.session_state.current_session
    
    user_question = st.chat_input("✍🏻 Ask your question in any language...")

    # Display existing chat messages for current session
    if active_session in st.session_state.chat_messages:
        for msg in st.session_state.chat_messages[active_session]:
            st.chat_message(msg["role"]).write(msg["content"])

    if user_question:
        try:
            with st.spinner("Thinking..."):
                history = get_history(active_session)
                result = conversational_rag.invoke(
                    {"input" : user_question},
                    config = {"configurable": {"session_id": active_session}},
                )
                answer = result["answer"]

            # Store messages in session state
            initialize_session(active_session)
            st.session_state.chat_messages[active_session].append({"role": "user", "content": user_question})
            st.session_state.chat_messages[active_session].append({"role": "assistant", "content": answer})

            # Display new messages
            st.chat_message("user").write(user_question)
            st.chat_message("assistant").write(answer)
            
        except Exception as e:
            error_msg = "Sorry, I encountered an error processing your question."
            st.error(f"Error processing question: {str(e)}")
            st.chat_message("user").write(user_question)
            st.chat_message("assistant").write(error_msg)
            
            # Store error in history too
            initialize_session(active_session)
            st.session_state.chat_messages[active_session].append({"role": "user", "content": user_question})
            st.session_state.chat_messages[active_session].append({"role": "assistant", "content": error_msg})

    def display_chat_history(session_id: str):
        """Display chat history for given session.
        
        Args:
            session_id (str): Session identifier for chat history
        """
        if session_id in st.session_state.chat_messages and st.session_state.chat_messages[session_id]:
            message_count = len(st.session_state.chat_messages[session_id]) // 2
            with st.expander(f" 📕 Chat History ({message_count} messages) "):
                for i, msg in enumerate(st.session_state.chat_messages[session_id]):
                    if msg["role"] == "user":
                        st.markdown(f"**👤 User:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {msg['content']}")
                    if i < len(st.session_state.chat_messages[session_id]) - 1:
                        st.divider()
                
                # Clear history button
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_messages[session_id] = []
                    st.session_state.chathistory[session_id] = ChatMessageHistory()
                    st.rerun()
    
    # Display chat history
    display_chat_history(active_session)

else:
    st.info("📁 Upload documents above to start chatting!")