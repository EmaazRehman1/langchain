import os
from dotenv import load_dotenv

from llm import API_KEY
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
API_KEY=os.getenv("OPEN_AI_API_KEY")
def get_docs():
    loader = WebBaseLoader('https://react.dev/learn/state-a-components-memory')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )

    splitDocs = text_splitter.split_documents(docs)

    return splitDocs

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorStore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorStore


def create_chain(vectorStore):
    model= ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        model="mistralai/mistral-7b-instruct:free",
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a helpful assistant. "
        "Answer ONLY with the provided context below. "
        "If the context is empty or does not contain the answer, respond with exactly: 'I don't know'.\n\n"
        "Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])



    # chain = prompt | model
    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retPrompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        {"role": "user", "content": "{input}"},
        {"role": "user", "content": "Given the above conversation, Generate a search query in order get information relevant to the conversation"},

    ])
    retriever = vectorStore.as_retriever()

    history_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retPrompt,
    )

    retrieval_chain = create_retrieval_chain(
        history_retriever, 
        document_chain,        
        )

    return retrieval_chain


def process_query(chain, question, chatHistory):
    response = chain.invoke({
        "input": question,
        "chat_history": chatHistory,
    })

    return response['answer']


if __name__ == "__main__": 
    docs = get_docs()
    vectorStore = create_vector_store(docs)
    chain = create_chain(vectorStore)

    chatHistory=[] 
    while True:
        inp=input("You: ")
        if inp=='0':
            break
        response = process_query(chain, inp, chatHistory)
        chatHistory.append(HumanMessage(content=inp))
        chatHistory.append(AIMessage(content=response))
        print(f"AI: {response}")
