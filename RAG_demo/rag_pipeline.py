import os

from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA

# --------------------------------
# Load PDF
# --------------------------------

loader = PyPDFLoader("company_policy.pdf")

documents = loader.load()

print(f"Pages Loaded : {len(documents)}")

# --------------------------------
# Split Text
# --------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Chunks Created : {len(chunks)}")

# --------------------------------
# Create Embeddings
# --------------------------------

embedding = OpenAIEmbeddings()

# --------------------------------
# Store in Vector Database
# --------------------------------

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="vector_db"
)

print("Vector Database Created")

# --------------------------------
# Retriever
# --------------------------------

retriever = vectordb.as_retriever(
    search_kwargs={"k":3}
)

# --------------------------------
# LLM
# --------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# --------------------------------
# QA Chain
# --------------------------------

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

print("\nRAG Ready!\n")

while True:

    question = input("Ask Question : ")

    if question.lower()=="exit":
        break

    answer = qa.invoke(question)

    print("\nAnswer:\n")

    print(answer["result"])

    print("-"*50)
