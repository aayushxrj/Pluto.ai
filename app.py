import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import chainlit as cl 
from chainlit.types import AskFileResponse

# For Approach 1
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

# For Approach 3
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


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
# def simple_retrieval(vectordb, message: cl.Message):
#     query = message.content

#     results = vectordb.similarity_search(query, k=5)
#     retrieved_documents = [result.page_content for result in results]

#     return retrieved_documents

@cl.step
def ChainOfThought():
    return "Not working yet"


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

    msg.content=f"Creating chunks for `{file.name}`..."
    await msg.update()

    # Store the data in form of embeddings
    vectordb = store_embeddings(chunks)

    msg.content = f"Creating embeddings for `{file.name}`. . ."
    await msg.update()

    # Approach 1 (low level approach)

    # SYSTEM_TEMPLATE = """
    # Answer the user's questions based on the below context. 
    # If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    # <context>
    # {context}
    # </context>
    # """

    # question_answering_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             SYSTEM_TEMPLATE,
    #         ),
    #         MessagesPlaceholder(variable_name="messages"),
            
    #     ]
    # )

    # document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    
    # def parse_retriever_input(params: Dict):
    #     return params["messages"][-1].content


    # chain = RunnablePassthrough.assign(
    #     context=parse_retriever_input | vectordb.as_retriever(k=5),
    # ).assign(
    #     answer=document_chain,
    # )


    # Approach 2 (high level approach)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k":5}),
        memory=memory,
        return_source_documents=True,
    )

    # debugging
    # if chain.memory is not None:
    #     print("The conversational retrieval chain has memory.")
    # else:
    #     print("The conversational retrieval chain does not have memory.")

    # Approach 3 (using load_qa_chain)

    # template="""  Answer the user's questions based on the below context and the previous chat history. 
    # If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    # <context>
    # {context}
    # </context>

    # <question>
    # {question}
    # </question>

    # <chat_history>
    # {chat_history}
    # </chat_history>

    # ANSWER :
    # """
    # memory = ConversationBufferMemory(memory_key="chat_history", input_key="question",max_len=50,return_messages=True)
    # prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)
 
    # chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)
    # cl.user_session.set("vectordb", vectordb)

    # Approach 4 (Approach 1 + memory)
    

    #COMMON TO ALL APPROACHES
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain) 


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    # Approach 1
    # response = chain.invoke({
    #     "messages": [
    #         HumanMessage(content=message.content)
    #     ],
    # })
    # await cl.Message(response["answer"]).send()

    # Approach 2
    response = await chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    print(response)                  # debugging
    answer = response["answer"]
    source_documents = response["source_documents"]
    text_elements = []
    unique_pages = set()

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            page_number = source_doc.metadata['page']
            page = f"Page {page_number}"
            text_element_content = source_doc.page_content
            # text_elements.append(cl.Text(content=text_element_content, name=source_name))
            if page not in unique_pages:
                unique_pages.add(page)
                text_elements.append(cl.Text(content=text_element_content, name=page))
            # text_elements.append(cl.Text(content=text_element_content, name=page))
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\n\nSources: {', '.join(source_names)}"
        else:
            answer += "\n\nNo sources found"

    # ChainOfThought()

    await cl.Message(content=answer, elements=text_elements).send()



    # Approach 3    (Sources link not working and old file cache issue)
    # vectordb = cl.user_session.get("vectordb")
    # docs = vectordb.similarity_search(query=message.content,k=5)

    # # what is the use of context when input_documents is acting as the context in load_qa_chain

    # chain_input={
    # "input_documents": docs,
    # "context":"This is contextless", 
    # "question":message.content
    # }

    # response = chain(chain_input, callbacks=[cl.AsyncLangchainCallbackHandler()])   # callbacks=[cl.AsyncLangchainCallbackHandler()] missing
    # print(response)                  # debugging

    # answer = response["output_text"]
    # # print source documents
    # source_documents = response["input_documents"]
    # text_elements = []
    # unique_pages = set()
    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         page_number = source_doc.metadata['page']
    #         page = f"Page {page_number}"
    #         text_element_content = source_doc.page_content
    #         if page not in unique_pages:
    #             unique_pages.add(page)
    #             text_elements.append(cl.Text(content=text_element_content, name=page))
    #     source_names = [text_el.name for text_el in text_elements]
    #     if source_names:
    #         answer += f"\n\nSources: {', '.join(source_names)}"
    #     else:
    #         answer += "\n\nNo sources found"

    # await cl.Message(content=answer).send()

    # Approach 4 (Approach 1 + memory)
