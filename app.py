import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import chainlit as cl 
from chainlit.types import AskFileResponse

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

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

    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = character_splitter.split_documents(pages)
    
    for i, doc in enumerate(splits):
        doc.metadata["source"] = f"source_{i}"
            
    print(f"Number of chunks: {len(splits)}")
    return splits


# Store the data in form of embeddings
def store_embeddings(chunks):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # making a vectordb
    vectordb = Chroma.from_documents(chunks, embedding_function)
    print(f"Size of vectordb: {vectordb._collection.count()}")
    return vectordb

# Retrieve data from the query
# def simple_retrieval(vectordb,message: cl.Message):
#     query = message.content   

#     results = vectordb.similarity_search(query, n_results=5)
#     retrieved_documents = results['documents'][0]
#     print(results)
#     print("\n\n")
#     print(f"Retrieved documents: {retrieved_documents}")
#     return results # debugging return 

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
    vectordb = store_embeddings(chunks)

    # Retrieve data from the query
    # retrived_documents = simple_retrieval(vectordb, cl.Message(content="What is the regression?"))

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    chain = ConversationalRetrievalChain.from_llm(
        ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229"),
        chain_type="stuff",
        retriever=vectordb.as_retriever(), 
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    



@cl.on_message
async def main(message: cl.Message):

    # chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    # cb = cl.AsyncLangchainCallbackHandler()
    # response = await chain.acall(message.content, callbacks=[cb])
    # print(response)

    tool()
    await cl.Message(content="Response not working yet").send()