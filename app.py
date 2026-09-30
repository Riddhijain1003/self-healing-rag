import os
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.set_page_config(page_title="Self-Healing RAG", page_icon="🔄", layout="centered")

@st.cache_resource(show_spinner="Initializing RAG Graph & Embeddings...")
def get_graph_app():
    # Importing workflow from main only once
    from main import app
    return app

st.title("🔄 Self-Healing RAG Assistant")
st.caption("Agentic RAG with real-time Critic evaluation, cyclical query rewriting, and graceful degradation.")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt
user_query = st.chat_input("Ask a question about company policies...")

if user_query:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    rag_app = get_graph_app()

    with st.chat_message("assistant"):
        status_container = st.status("Evaluating through Self-Healing Graph...", expanded=True)
        try:
            status_container.write("🔍 Retrieving documents & generating candidate answer...")
            
            initial_state = {
                "question": user_query,
                "retry_count": 0
            }
            
            result = rag_app.invoke(initial_state)
            generation = result.get("generation", "No response generated.")
            
            status_container.update(label="Complete!", state="complete", expanded=False)
            st.markdown(generation)
            st.session_state.messages.append({"role": "assistant", "content": generation})
            
        except Exception as e:
            status_container.update(label="Failed", state="error")
            st.error(f"Error executing graph: {e}")
