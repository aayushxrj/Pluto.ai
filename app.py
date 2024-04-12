import os
# from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import chainlit as cl 
from chainlit.types import AskFileResponse


welcome_message = """Welcome to the Pluto.ai ! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

# Function to process the file
def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

        loader = Loader(file.path)
        pages = loader.load()
        # print(pages[0].page_content)
        return pages


# Split the data into chunks
def split_into_chunks(file: AskFileResponse):
    pages = process_file(file)


    data = [page.page_content for page in pages]

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text('\n\n'.join(data))        
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    print(f"\nTotal chunks: {len(token_split_texts)}")
    return token_split_texts

# Store the data in form of embeddings
def store_embeddings(chunks):
    embedding_function = SentenceTransformerEmbeddingFunction()

    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("doc_chroma_collection", embedding_function=embedding_function)

    ids = [str(i) for i in range(len(chunks))]

    chroma_collection.add(ids=ids, documents=chunks)
    print(f"Size of chroma_collection: {chroma_collection.count()}")
    return chroma_collection
    


# Retrieve data from the query
def simple_retrieval(chroma_collection):
    query = "What is Microsoft?"

    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]
    print(f"Retrieved documents: {retrieved_documents}")

@cl.step
def tool():
    return "Chain of thought not working yet"


@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Pluto",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    ).send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    # Process the file and split into chunks
    chunks = split_into_chunks(file)

    # Store the data in form of embeddings
    chroma_collection = store_embeddings(chunks)
    
    # Retrieve data from the query
    simple_retrieval(chroma_collection)

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    



@cl.on_message
async def main(message: cl.Message):
    tool()
    await cl.Message(content="Response not working yet").send()