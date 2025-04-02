import os
import fitz  # PyMuPDF
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
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
embeddings = OpenAIEmbeddings()

def create_vector_store(chunks):
    return FAISS.from_documents(chunks, embeddings)

# Step 4: Store in Vector DB
def store_vectors(vector_store, path="faiss_index"):
    vector_store.save_local(path)

def load_vectors(path="faiss_index"):
    return FAISS.load_local(path, embeddings)

# Step 5: Querying the Vector Store
def retrieve_documents(query, vector_store, k=3):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)

 # Initializing llm from groq

 llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
    api_key = 
)

# Step 6: Retrieve Context for Generation
def generate_answer(query, vector_store, model="gpt-3.5-turbo-instruct"):
    retriever = vector_store.as_retriever()
    qa_chain = create_retrieval_chain(llm=OpenAI(model_name=model), chain_type="stuff", retriever=retriever)
    return qa_chain.run(query)

# Example usage
folder_path = "./documents"  # Change this to your folder path
documents = load_documents(folder_path)
chunks = split_documents(documents)
vector_store = create_vector_store(chunks)
store_vectors(vector_store)

# Load and Query
db = load_vectors()
query = "What is Retrieval-Augmented Generation?"
response = generate_answer(query, db)
print(response)

