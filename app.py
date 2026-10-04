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
    # Importing from main only once (builds the index and compiles the graph)
    from main import app
    return app


def describe(node: str, update: dict) -> str:
    """Turns one LangGraph node update into a human-readable trace line."""
    if node == "retrieve":
        return f"🔍 **Retrieve**: fetched {len(update.get('documents', []))} chunks"
    if node == "generate":
        return f"✍️ **Generate**: “{update.get('generation', '')}”"
    if node == "grade":
        return (
            f"🧪 **Critic**: grounded = {update.get('grounded')}, "
            f"answered = {update.get('answered')}, rejections so far = {update.get('retry_count')}"
        )
    if node == "rewrite_query":
        return f"🔄 **Rewrite query**: “{update.get('search_query', '')}”"
    if node == "fallback_response":
        return "🛟 **Fallback**: no reliable answer found, returning a safe response"
    return f"{node}"


st.title("🔄 Self-Healing RAG Assistant")
st.caption("Agentic RAG with real-time Critic evaluation, cyclical query rewriting, and graceful degradation.")

with st.sidebar:
    st.subheader("Try these")
    st.markdown(
        "- How much does express delivery cost?\n"
        "- My phone arrived broken, what now?\n"
        "- Can I get a refund on a gift card?\n"
        "- Do you ship internationally?"
    )
    st.caption("Each question is answered independently (no chat memory).")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("How the pipeline got here"):
                for line in msg["trace"]:
                    st.markdown(line)

# User prompt
user_query = st.chat_input("Ask a question about company policies...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    rag_app = get_graph_app()

    with st.chat_message("assistant"):
        status = st.status("Running self-healing graph...", expanded=True)
        trace = []
        final_state = {}
        try:
            initial_state = {
                "question": user_query,
                "search_query": user_query,
                "retry_count": 0,
            }

            # stream_mode="updates" yields {node_name: state_update} as each node finishes
            for step in rag_app.stream(initial_state, stream_mode="updates"):
                for node, update in step.items():
                    final_state.update(update)
                    line = describe(node, update)
                    trace.append(line)
                    status.write(line)

            generation = final_state.get("generation", "No response generated.")
            rejections = final_state.get("retry_count", 0)
            status.update(label=f"Complete · critic rejected {rejections}×", state="complete", expanded=False)

            st.markdown(generation)
            st.session_state.messages.append(
                {"role": "assistant", "content": generation, "trace": trace}
            )

        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Error executing graph: {e}")