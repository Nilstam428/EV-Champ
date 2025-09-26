from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

VECTOR_STORE_PATH = "./app/vector_store" # Path relative to the project root
EMBEDDING_MODEL = OpenAIEmbeddings()

# 1. Connect to the existing vector store
vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=EMBEDDING_MODEL)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 most relevant chunks

# 2. Create a prompt template optimized for your EV use case
template = """
You are an expert assistant for electric vehicle (EV) owners.
Answer the user's question based ONLY on the following context from user manuals and FAQs.
If the information is not in the context, say you don't have that information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = PromptTemplate.from_template(template)

# 3. Set up the Language Model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# 4. Define the RAG chain to orchestrate the process
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_rag_response(query: str) -> str:
    """
    Invokes the RAG chain with a user query.
    """
    return rag_chain.invoke(query)