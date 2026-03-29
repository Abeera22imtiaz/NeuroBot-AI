from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# import mlflow 

# Step 1: Load raw PDF
DATA_PATH = 'data/'

def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load() 
    return documents

documents = load_pdf_files(data=DATA_PATH)
#print('Total PDF pages loaded:', len(documents))

# Step 2: Create Chunks (Direct 500 size)
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
#print('Total Text Chunks created (Size 500):', len(final_chunks))

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  
    )
    return embedding_model

embedding_model=get_embedding_model()
'''
all-MiniLM-L6-v2
This is a sentence-transformers model: It maps sentences & paragraphs
 to a 384 dimensional dense vector space and 
 can be used for tasks like clustering or semantic search.
'''


# --- MLflow Experiment Function (Commented Out for now) ---
# def experiment_chunks(documents, size, overlap):
#     with mlflow.start_run(run_name=f"ChunkSize_{size}"):
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
#         chunks = text_splitter.split_documents(documents)
#         mlflow.log_param("chunk_size", size)
#         mlflow.log_param("chunk_overlap", overlap)
#         mlflow.log_metric("num_chunks", len(chunks))
#         print(f"✅ Run Complete: Size {size}, Chunks: {len(chunks)}")
#         return chunks


    # --- MLflow Section (Optional/Commented) ---
    # mlflow.set_experiment("NeuralGuide_Chunking_Test")
    # sizes_to_test = [400, 500, 600, 800]
    # for s in sizes_to_test:
    #     experiment_chunks(documents, size=s, overlap=50)

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)