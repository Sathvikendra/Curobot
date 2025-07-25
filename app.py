from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.helper import download_embeddings,rag_or_gemini,log_chat
from src.prompt import *
from flask import Flask,render_template,jsonify,request
import sys

app=Flask(__name__)
load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embeddings=download_embeddings()

pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name="curobot"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name,dimension=384,metric="cosine",
                    spec=ServerlessSpec(cloud="aws",region="us-east-1"))
index=pc.Index(index_name)

docsearch=PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings)

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

genai.configure(api_key=GOOGLE_API_KEY)
model=ChatGoogleGenerativeAI(model="gemini-2.5-pro",google_api_key=GOOGLE_API_KEY)

chat_history=[]

prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("user","Context : {context}\n\n Question : {input}")
])

question_answer_chain=create_stuff_documents_chain(model,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

@app.before_request
def clear_chat_on_reload():
    global chat_history
    if request.endpoint == 'index':
        chat_history=[]

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    query=msg
    print(query)
    result=rag_or_gemini(model,retriever,query,chat_history,rag_chain)
    if isinstance(result, dict):
        response_text = result["result"]
    elif isinstance(result, str):
        response_text = result
    else:
        response_text = "Sorry, something went wrong."
    try:
        print("Response : ",response_text)
    except OSError:
        sys.stdout.write(("Response : " + str(response_text) + "\n").encode("utf-8", "ignore").decode("utf-8"))
    log_chat(query,response_text,chat_history)
    return str(response_text)

if __name__=='__main__':
    app.run(debug=True)