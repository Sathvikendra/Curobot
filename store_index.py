from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf,text_split,download_embeddings


load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

data=load_pdf(data='data/')
chunk=text_split(data)
embeddings=download_embeddings()

pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name="curobot"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name,dimension=384,metric="cosine",
                    spec=ServerlessSpec(cloud="aws",region="us-east-1"))
index=pc.Index(index_name)

docsearch=PineconeVectorStore.from_documents(documents=chunk,embedding=embeddings,index_name=index_name)