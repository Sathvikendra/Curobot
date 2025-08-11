from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from flask import Flask, render_template, request
from src.helper import rag_or_gemini, log_chat
from src.prompt import system_prompt

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

# ------------------------------
# GLOBALS (only loaded once)
# ------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
pinecone_client = None
retriever = None
model = None
rag_chain = None
chat_history = []

def init_pinecone_and_chain():
    """Initialize Pinecone, retriever, and RAG chain only once."""
    global pinecone_client, retriever, model, rag_chain

    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

        index_name = "curobot"
        if index_name not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        # Reuse the global embeddings here
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=GOOGLE_API_KEY
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context : {context}\n\n Question : {input}")
        ])

        question_answer_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.before_request
def clear_chat_on_reload():
    global chat_history
    if request.endpoint == 'index':
        chat_history = []

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global chat_history
    init_pinecone_and_chain()

    msg = request.form["msg"]
    query = msg.strip()

    result = rag_or_gemini(model, retriever, query, chat_history, rag_chain)

    if isinstance(result, dict):
        response_text = result.get("result", "")
    elif isinstance(result, str):
        response_text = result
    else:
        response_text = "Sorry, something went wrong."

    log_chat(query, response_text, chat_history)
    return str(response_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
