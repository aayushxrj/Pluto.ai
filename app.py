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

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage


ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")

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
def simple_retrieval(vectordb, message: cl.Message):
    query = message.content

    results = vectordb.similarity_search(query, k=5)
    retrieved_documents = [result.page_content for result in results]

    return retrieved_documents


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
    # print(retrived_documents)
    
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. 
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    
    def parse_retriever_input(params: Dict):
        return params["messages"][-1].content


    chain = RunnablePassthrough.assign(
        context=parse_retriever_input | vectordb.as_retriever(k=5),
    ).assign(
        answer=document_chain,
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    
    # response = await chain.acall(message.content, 
    #                                  callbacks=[
    #                                      cl.AsyncLangchainCallbackHandler()])
    response = chain.invoke(
        {"messages": [
            HumanMessage(content=message.content)
        ],
    }
    )
    tool()
    await cl.Message(response["answer"]).send()