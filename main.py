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
    question: str
    documents: List[str]
    generation: str
    retry_count: int

# --- 3. Node Definitions ---
def retrieve(state: GraphState):
    print("\n--- [NODE] RETRIEVING DOCUMENTS ---")
    question = state["question"]
    docs = retriever.invoke(question)
    doc_strings = [d.page_content for d in docs]
    print(f"Retrieved {len(doc_strings)} chunks for question: '{question}'")
    return {"documents": doc_strings, "retry_count": state.get("retry_count", 0)}

def generate(state: GraphState):
    print("\n--- [NODE] GENERATING ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks.\n"
        "Answer the question using ONLY the facts explicitly provided in the context below.\n"
        "If the answer is not mentioned in the context, respond strictly with 'I don't know.'\n"
        "Do not extrapolate, assume, or provide answers for digital goods unless specifically stated.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    chain = prompt | llm
    response = chain.invoke({"context": "\n\n".join(documents), "question": question})
    return {"generation": response.content.strip()}

def rewrite_query(state: GraphState):
    print("\n--- [NODE] REWRITING QUERY (CRITIC REJECTED) ---")
    question = state["question"]
    prompt = ChatPromptTemplate.from_template(
        "You are reformulating a search query that failed to retrieve relevant documents.\n"
        "Initial question: {question}\n"
        "Provide a different, keyword-focused search query. Output ONLY the query:"
    )
    chain = prompt | llm
    better_query = chain.invoke({"question": question}).content.strip()
    new_retry = state["retry_count"] + 1
    print(f"Reformulated Query (Attempt {new_retry}): '{better_query}'")
    return {"question": better_query, "retry_count": new_retry}

def fallback_response(state: GraphState):
    print("\n--- [NODE] FALLBACK TRIGGERED ---")
    return {"generation": "I don't have enough information in the provided documents to answer your question."}

# --- 4. Critic & Conditional Routing ---
def grade_generation(state: GraphState):
    print("\n--- [CRITIC] AUDITING GROUNDEDNESS ---")
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state["retry_count"]
    
    critic_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an auditor verifying whether an answer is grounded in facts.\n"
         "Does the candidate answer invent details, extrapolate rules, or assume anything NOT directly stated in the context?\n"
         "If the answer invents or extrapolates anything, reply with 'NO'.\n"
         "If the answer is fully grounded or says 'I don't know', reply with 'YES'.\n"
         "Respond with ONLY 'YES' or 'NO'."),
        ("human", "Context:\n{context}\n\nCandidate Answer:\n{generation}")
    ])
    
    critic_chain = critic_prompt | llm
    verdict = critic_chain.invoke({
        "context": "\n\n".join(documents),
        "generation": generation
    }).content.strip().upper()
    
    print(f"Critic Verdict: {verdict} | Retry Count: {retry_count}")

    # Trap detection: reject if hallucinated or if it admits it doesn't know
    if "YES" in verdict and "i don't know" not in generation.lower():
        print("-> Verdict Accepted: Answer is valid and grounded.")
        return "useful"
    elif retry_count < 2:
        print("-> Verdict Rejected: Triggering rewrite loop...")
        return "retry"
    else:
        print("-> Max retries reached: Routing to fallback.")
        return "fallback"

# --- 5. Graph Assembly ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("fallback_response", fallback_response)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")

workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "useful": END,
        "retry": "rewrite_query",
        "fallback": "fallback_response",
    }
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


# --- 6. Execution ---
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
            "retry_count": 0
        }
        
        final_output = app.invoke(initial_state)
        print("\n--- FINAL ANSWER ---")
        print(final_output["generation"])
        print("-" * 30 + "\n")
