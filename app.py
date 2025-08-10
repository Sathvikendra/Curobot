from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
from functools import lru_cache

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

from src.helper import download_embeddings, rag_or_gemini, log_chat
from src.prompt import system_prompt

import sys

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure env vars are available to libraries
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

chat_history = []

@lru_cache(maxsize=1)
def get_rag_components():
    """
    Loads and caches embeddings, Pinecone retriever, model, and chain.
    This function is only called once per server lifetime.
    """
    embeddings = download_embeddings()  # Use a lightweight embedding model here

    # Connect to existing Pinecone index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "curobot"  # Must already exist (create locally, not here)

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    genai.configure(api_key=GOOGLE_API_KEY)
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

    return model, retriever, rag_chain


@app.before_request
def clear_chat_on_reload():
    global chat_history
    if request.endpoint == 'index':
        chat_history = []


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    query = msg
    print(query)

    model, retriever, rag_chain = get_rag_components()

    result = rag_or_gemini(model, retriever, query, chat_history, rag_chain)

    if isinstance(result, dict):
        response_text = result.get("result", "")
    elif isinstance(result, str):
        response_text = result
    else:
        response_text = "Sorry, something went wrong."

    try:
        print("Response : ", response_text)
    except OSError:
        sys.stdout.write(("Response : " + str(response_text) + "\n").encode("utf-8", "ignore").decode("utf-8"))

    log_chat(query, response_text, chat_history)
    return str(response_text)


if __name__ == '__main__':
    # In Render, use gunicorn instead of running this directly
    app.run(host="0.0.0.0", port=5000, debug=False)
