import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google import OpenAIEmbeddings

# Define paths
DATA_PATH = "../app/data"
VECTOR_STORE_PATH = "../app/vector_store"

def create_vector_store():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and stores them in a ChromaDB persistent vector store.
    """
    print("Loading EV documents...")
    # Using DirectoryLoader to load multiple file types
    loader = DirectoryLoader(DATA_PATH, glob="**/*", loader_map={
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader
    }, show_progress=True)
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Creating embeddings and storing in ChromaDB at '{VECTOR_STORE_PATH}'...")
    # Using OpenAI for embeddings. Ensure OPENAI_API_KEY is set in your environment.
    embeddings = OpenAIEmbeddings()

    # Create a new ChromaDB store from the documents
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )

    print(f"Successfully created vector store with {vectorstore._collection.count()} documents.")

if __name__ == "__main__":
    create_vector_store()