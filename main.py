import os
import warnings

# Suppress HuggingFace and LangChain deprecation noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

load_dotenv()

MAX_RETRIES = 2  # total healing attempts after the first try

# --- 1. Document Loading & Indexing ---
loader = TextLoader("docs/company.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# --- 2. State Definition ---
class GraphState(TypedDict):
    question: str        # ORIGINAL user question. Never modified.
    search_query: str    # What we send to the retriever. Rewrite node changes only this.
    documents: List[str]
    generation: str
    retry_count: int     # how many times the critic has rejected so far
    grounded: bool       # critic verdict 1: no invented facts?
    answered: bool       # critic verdict 2: did it actually answer the question?


# --- 3. Critic output schema (structured output, no fragile "YES" in text) ---
class Grade(BaseModel):
    grounded: bool = Field(
        description=(
            "True only if EVERY claim in the answer is directly supported by the context. "
            "An answer like 'I don't know' invents nothing, so it is grounded."
        )
    )
    answered: bool = Field(
        description=(
            "True only if the answer actually addresses the question. "
            "False if it is a refusal or says it doesn't know."
        )
    )


critic_llm = llm.with_structured_output(Grade)


# --- 4. Node Definitions ---
def retrieve(state: GraphState):
    print("\n--- [NODE] RETRIEVING DOCUMENTS ---")
    query = state.get("search_query") or state["question"]
    found = retriever.invoke(query)
    doc_strings = [d.page_content for d in found]
    print(f"Retrieved {len(doc_strings)} chunks for query: '{query}'")
    return {"documents": doc_strings}


def generate(state: GraphState):
    print("\n--- [NODE] GENERATING ANSWER ---")
    question = state["question"]  # always the ORIGINAL question
    documents = state["documents"]

    # If the critic previously caught a hallucination, be stricter this time.
    strict_note = ""
    if state.get("retry_count", 0) > 0 and state.get("grounded") is False:
        strict_note = (
            "WARNING: your previous answer contained claims not found in the context. "
            "State only facts that appear word-for-word in the context.\n"
        )

    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks.\n"
        "Answer the question using ONLY the facts explicitly provided in the context below.\n"
        "If the answer is not mentioned in the context, respond strictly with \"I don't know.\"\n"
        "Do not extrapolate or assume anything that is not explicitly stated.\n"
        "{strict_note}\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    chain = prompt | llm
    response = chain.invoke({
        "context": "\n\n".join(documents),
        "question": question,
        "strict_note": strict_note,
    })
    return {"generation": response.content.strip()}


def grade(state: GraphState):
    """Critic node: audits the answer and does the retry bookkeeping."""
    print("\n--- [CRITIC] AUDITING ANSWER ---")

    critic_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict auditor of a question-answering system.\n"
         "Judge two things:\n"
         "1. grounded: is every claim in the answer directly supported by the context? "
         "Inventing, extrapolating, or assuming anything NOT stated in the context means false.\n"
         "2. answered: does the answer actually address the question? "
         "A refusal or 'I don't know' means false."),
        ("human",
         "Context:\n{context}\n\nQuestion:\n{question}\n\nCandidate Answer:\n{generation}"),
    ])

    result: Grade = (critic_prompt | critic_llm).invoke({
        "context": "\n\n".join(state["documents"]),
        "question": state["question"],
        "generation": state["generation"],
    })

    accepted = result.grounded and result.answered
    new_retry = state["retry_count"] + (0 if accepted else 1)
    print(f"Critic -> grounded={result.grounded}, answered={result.answered} | retry_count={new_retry}")

    return {
        "grounded": result.grounded,
        "answered": result.answered,
        "retry_count": new_retry,
    }


def rewrite_query(state: GraphState):
    print("\n--- [NODE] REWRITING SEARCH QUERY ---")
    prompt = ChatPromptTemplate.from_template(
    "Original question: {question}\n"
    "Previous search query: {previous}\n"
    "Rewrite it as ONE short natural-language query (max 10 words) with different wording.\n"
    "Rules: no quotes, no OR/AND, no site: or other search operators, "
    "no new facts or names that are not in the original question.\n"
    "Output ONLY the query:"
)
    chain = prompt | llm
    better = chain.invoke({
        "question": state["question"],
        "previous": state.get("search_query") or state["question"],
    }).content.strip()
    print(f"New search query (retry {state['retry_count']}): '{better}'")
    return {"search_query": better}  # question stays untouched


def fallback_response(state: GraphState):
    print("\n--- [NODE] FALLBACK TRIGGERED ---")
    return {
        "generation": "I don't have enough information in the provided documents to answer your question."
    }


# --- 5. Routing (pure decision, no state changes here) ---
def route_after_grade(state: GraphState):
    if state["grounded"] and state["answered"]:
        print("-> Accepted: answer is grounded and answers the question.")
        return "useful"
    if state["retry_count"] > MAX_RETRIES:
        print("-> Max retries reached: routing to fallback.")
        return "fallback"
    if not state["grounded"]:
        print("-> Hallucination: regenerating from the same documents.")
        return "regenerate"
    print("-> Could not answer: rewriting the search query.")
    return "rewrite"


# --- 6. Graph Assembly ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade", grade)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("fallback_response", fallback_response)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade")

workflow.add_conditional_edges(
    "grade",
    route_after_grade,
    {
        "useful": END,
        "regenerate": "generate",       # hallucination -> same docs, stricter prompt
        "rewrite": "rewrite_query",     # retrieval problem -> better query
        "fallback": "fallback_response",
    },
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("fallback_response", END)

app = workflow.compile()

try:
    with open("graph_architecture.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("Saved workflow diagram as graph_architecture.png")
except Exception as e:
    print(f"Could not render diagram automatically: {e}")


# --- 7. Execution ---
if __name__ == "__main__":
    print("\n=== Self-Healing RAG Pipeline Initialized ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("Ask a question: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            continue

        initial_state = {
            "question": user_query,
            "search_query": user_query,
            "retry_count": 0,
        }

        final_output = app.invoke(initial_state)
        print("\n--- FINAL ANSWER ---")
        print(final_output["generation"])
        print("-" * 30 + "\n")