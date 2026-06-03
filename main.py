from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Step 1: Document load karo
loader = TextLoader("docs/company.txt")
docs = loader.load()

# Step 2: Chunks banao
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

# Step 3: Embed karo aur FAISS mein store karo
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 4: LLM
llm = ChatGroq(model="llama3-8b-8192")

# Step 5: Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context: {context}

Question: {question}
""")

# Step 6: Chain banao
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Step 7: Poochho!
question = "What is the return policy?"
answer = chain.invoke(question)
print("Answer:", answer.content)
