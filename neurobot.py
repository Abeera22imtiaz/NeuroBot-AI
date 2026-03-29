import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstores():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(HF_TOKEN):
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.1,
        max_new_tokens=512,
        timeout=300,
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

def main():
    st.title('Ask Chatbot!')

    # Initialize messages list if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history correctly
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input('Enter your prompt here!')
    if prompt:
        # Show user message
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

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

        # Token handling
        if "HF_TOKEN" in st.secrets:
            HF_TOKEN = st.secrets["HF_TOKEN"]
        else:
            from dotenv import load_dotenv
            load_dotenv()
            HF_TOKEN = os.getenv("HF_TOKEN")

        try:
            vectorstore = get_vectorstores()
            if vectorstore is None:
                st.error('Failed to load vector store')
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            source_document = response.get("source_documents", [])

            # Result formatting
            result_to_show = f"{result}\n\n---\n**Sources Used:**\n"

            if source_document:
                for i, doc in enumerate(source_document):
                    source_name = doc.metadata.get('source', 'Unknown Book')
                    page_num = doc.metadata.get('page', 'N/A')
                    result_to_show += f"[{i+1}] {source_name} (Page {page_num})\n"
            else:
                result_to_show += "No specific sources found."

            # Bot response
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main()