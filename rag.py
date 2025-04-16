import os
import fitz  # PyMuPDF
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # Using FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # Importing ChatGroq for Groq model
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from dotenv import load_dotenv  # Importing to load environment variables

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Ensure your .env file has this key

# Step 1: Load Documents
def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        docs.extend(loader.load())
    return docs

# Step 2: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def split_documents(docs):
    return text_splitter.split_documents(docs)

# Step 3: Embed the Text
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")  # Using FastEmbedEmbeddings

def create_vector_store(chunks):
    return  FAISS.from_documents(chunks, embeddings)

# Step 4: Store in Vector DB
def store_vectors(vector_store, path="faiss_index"):
    vector_store.save_local(path)

def load_vectors(path="faiss_index"):
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# Step 5: Querying the Vector Store
def retrieve_documents(query, vector_store, k=3):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)

# Initializing llm from Groq
llm = ChatGroq(
    model_name="llama3-8b-8192",  # Adjusted model name
    temperature=0.7,
    api_key=GROQ_API_KEY  # Pass the API key to the Groq model
)

# Step 6: Retrieve Context for Generation
def generate_answer(query, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Adjust k for the number of documents to retrieve
    retrieved_docs = retriever.get_relevant_documents(query)  # Retrieve documents based on semantic search
    context = " ".join([doc.page_content for doc in retrieved_docs])  # Combine the content of retrieved documents

    # Create a prompt that includes the context
    prompt = f"""
    You are a question answering bot. You will be given a QUESTION and a set of paragraphs in the CONTENT section. 
    You need to answer the question using the text present in the CONTENT section. 
    If the answer is not present in the CONTENT text then reply `I don't have answer to the question`

    CONTENT: {context}
    QUESTION: {query}
    """

    print("context : ", context)
    # Use the Groq model to generate an answer
    llm_answer = llm.invoke(prompt)
    return llm_answer

# Example usage
folder_path = "./documents"  # Change this to your folder path
documents = load_documents(folder_path)
chunks = split_documents(documents)
vector_store = create_vector_store(chunks)
store_vectors(vector_store)

# Load and Query
db = load_vectors()
query = input("Enter your prompt : ")
response = generate_answer(query, db)
print(response)
