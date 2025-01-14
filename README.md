# ChatBot for Freelancing Labor Workers

## Overview

This project is a chatbot application designed to provide users with detailed information about freelancing labor workers available in a MongoDB database. The chatbot uses LangChain and FastAPI to create a conversational experience where users can:

- Query information about workers.
- Identify suitable workers for specific tasks based on their skills, location, and availability.
- Retrieve past chat session history.

The application leverages FastAPI for deployment and MongoDB for storing worker details. The LangChain framework handles context-aware conversations and retrieval-augmented generation (RAG) for providing accurate responses.

---

## Features

1. **Retrieve Worker Information**:
   - Query details like name, contact, skills, availability, and more.

2. **Context-Aware Conversations**:
   - Uses LangChain's history-aware retriever to maintain context during chats.

3. **Natural Language Understanding**:
   - Reformulates user queries into standalone, self-contained questions for better retrieval.

4. **Worker Matching**:
   - Suggests workers based on the described problem, their skills, location, and availability.

5. **Session History Management**:
   - Stores and retrieves chat history for personalized and continuous interactions.

6. **FastAPI Deployment**:
   - Provides RESTful APIs for interacting with the chatbot.

---



## How It Works

1. **Data Retrieval**:
   - Worker details are fetched from the MongoDB collection.
   - Data is formatted into LangChain-compatible `Document` objects.

2. **Text Splitting and Embeddings**:
   - Documents are split into chunks using `RecursiveCharacterTextSplitter`.
   - Embeddings are generated using Google Generative AI.

3. **Contextual Retrieval**:
   - User queries are reformulated into standalone questions using a retriever prompt.
   - Relevant worker information is retrieved based on the query.

4. **Worker Matching**:
   - The chatbot matches workers to user requirements based on skills, location, and availability.

5. **FastAPI Integration**:
   - APIs enable seamless interaction with the chatbot for real-time information retrieval.

---

## Example Usage

![Screenshot from 2025-01-14 05-40-56](https://github.com/user-attachments/assets/d5002cf5-c2ca-4f3d-aa11-ea5391e7bdcd)

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes. Your contributions are highly appreciated!
