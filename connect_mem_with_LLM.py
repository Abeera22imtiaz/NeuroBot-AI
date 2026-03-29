import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import time
# 1. Load Token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Setup LLM (FIX: wrap inside ChatHuggingFace)
def load_llm():
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.1,
        max_new_tokens=512,
        timeout=300,
        # task optional hai ab
    )

    chat_model = ChatHuggingFace(llm=llm)  
    return chat_model

# 3. Load Vector Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# 4. Prompt Template
# 4. Prompt Template (FIXED)
PROMPT_TEMPLATE = """You are NeuroBot, an Advanced AI Career & Technical Mentor.
You have access to a library of 4 core books:
1. Nielsen: Basic Neural Network Theory.
2. Deisenroth: Mathematical foundations.
3. D2L: Practical implementation.
4. Burkov: Real-world ML Engineering.

Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

NeuroBot's Answer:"""

# Ensure input_variables has both context and question
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# 5. Create Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),   
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt}
)

# 6. Run
if __name__ == "__main__":
    print("\n--- Chatbot Ready!")
    query = input("Enter your Query: ")
    
    try:
        # 1. Start time 
        start_time = time.time()
        
        # 2. Run chain
        response = qa_chain.invoke({"query": query})
        
        # 3. End time 
        end_time = time.time()
        
        # 4. Total time calculate 
        elapsed_time = end_time - start_time
        
        print("\n" + "="*50)
        print(f"⏱️ Response Time: {elapsed_time:.2f} seconds")
        print("="*50)
        
        print("\nYour Answer:\n", response["result"])
        
        # Sources 
        print("\nSource Documents:")
        for doc in response["source_documents"]:
            print(f"- Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}")
            
    except Exception as e:
        print(f"\nError: {e}")