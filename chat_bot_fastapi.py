from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
import os
from dot_env import load_env()

load_env()

os.getenviron["GOOGLE_API_KEYS"]= "AIzaSyBoHKr5qJ54IixWyCmgg4cgUop0KezakGw"

# Import your chatbot-related setup
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.docstore.document import Document

# MongoDB connection
client = MongoClient("mongodb+srv://hirenngood:dPJFE3VO8TULNgDF@wisdomwise.langpsw.mongodb.net/")
db = client["Karyamitra"]
collection = db["serviceprovider"]

documents = list(collection.find())

# Step 2: Format MongoDB data into LangChain Documents
formatted_docs = []
for doc in documents:
    # Construct content as a string
    content = (
        f"Name: {doc.get('firstName', '')} {doc.get('middleName', '')} {doc.get('lastName', '')}\n"
        f"Gender: {doc.get('gender', '')}\n"
        f"Date of Birth: {doc.get('dob', '')}\n"
        f"Email: {doc.get('email', '')}\n"
        f"Mobile No: {doc.get('mobileNo', '')}\n"
        f"Profile Image: {doc.get('profileImage', '')}\n"
        f"Is Mobile Verified: {doc.get('isMobileNoVerified', False)}\n"
        f"Is Verified: {doc.get('isVerified', False)}\n"
        f"Address: {doc.get('address', [])}\n"
        f"Availability: {doc.get('availabilityTime', [])}\n"
        f"Work Location: {doc.get('workLocation', {})}\n"
        f"Verification Status: {doc.get('verificationStatus', '')}\n"
        f"Work Images/Videos: {doc.get('workImgVideo', [])}\n"
        f"Work History: {doc.get('workHistory', [])}\n"
        f"Reviews: {doc.get('reviews', [])}\n"
        f"Profile: {doc.get('profile', {})}\n"
        f"Created At: {doc.get('createdAt', '')}\n"
        f"Updated At: {doc.get('updatedAt', '')}\n"
    )
    # Create a Document object
    formatted_docs.append(Document(page_content=content))

# Text splitting and embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
split_docs = text_splitter.split_documents(formatted_docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)

retriever = vectorstore.as_retriever()

retriever_prompt = (
    """Given a chat history and the latest user question which references the information about workers provided in the context, 
reformulate the question to make it standalone and self-contained. 
The reformulated question should focus specifically on the details of workers or their attributes. 
Do NOT answer the question, just rephrase it as needed to ensure clarity and completeness."""
)
     
system_prompt = (
    """You are an assistant specialized in providing answers about workers based on the given lists and information.
When the user provides a scenario or describes a faulty incident in their daily life, use the retrieved details about workers to identify the most suitable worker to address the issue.
Match the worker's skills, availability, and location to the problem described by the user. If no suitable worker can be identified from the provided information, respond with "I don't know based on the available data."

Example:
User Input: "My kitchen sink is leaking, and I need a plumber who can come today to fix it."
Assistant Response: "The most suitable worker is John Doe, a plumber available today, located near your area. Contact: 9876543210."

    "\n\n"
    "{context}"
    """
)

model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key="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", temperature=0)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Session history store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# FastAPI app
app = FastAPI()

# Request and response schemas
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = conversational_rag_chain.invoke(
            {"input": request.message},
            config={
                "configurable": {"session_id": request.session_id}
            }
        )
        return ChatResponse(answer=response["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session_messages(session_id: str):
    # Get the session history
    if session_id not in store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_history = store[session_id]
    messages = session_history.messages

    # Format the messages for response
    formatted_messages = [
        {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
        for msg in messages
    ]

    return {"session_id": session_id, "messages": formatted_messages}
