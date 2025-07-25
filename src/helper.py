from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from datetime import datetime
from src.prompt import *

def load_pdf(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    docs=loader.load()
    return docs

def text_split(data):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    chunk=splitter.split_documents(data)
    return chunk

def download_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def log_chat(query, answer, chat_history):
    chat_history.append({
        "timestamp": datetime.now().isoformat(),
        "user": query,
        "bot": answer
    })

def format_history_for_context(history):
    return "\n".join(f"User : {msg['user']}\nBot : {msg['bot']}" for msg in history)

def gemini_response(model,query,context=None):
    response=model.invoke("\n\n".join([fallback_prompt,f"User query : {query}",f"Additional Context : {context}"]))
    return response.content

def rag_or_gemini(model,retriever,query,chat_history,rag_chain,threshold=0.5):
    retrieved_docs=retriever.get_relevant_documents(query)
    context_text_db="\n\n".join([doc.page_content for doc in retrieved_docs])
    context_text_chat = format_history_for_context(chat_history)
    context_chosen=context_text_chat if chat_history else context_text_db
    if not chat_history and retrieved_docs:
        try:
            result=rag_chain.invoke({"input":query})
            return result["answer"]
        except Exception as e:
            print("RAG failed. Falling back to Gemini")
    
    return gemini_response(model,query,context=context_chosen)