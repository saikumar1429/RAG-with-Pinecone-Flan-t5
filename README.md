# Retrieval-Augmented Generation (RAG) with Pinecone and Flan-T5

This project implements a Retrieval-Augmented Generation (RAG) system using Pinecone for vector search, SentenceTransformers for embeddings, and Google's Flan-T5 model for answer generation. The application is built with Streamlit and provides an interactive interface for asking questions based on stored contextual data.

---

## Project Overview

The system combines semantic search with language generation to provide context-aware answers.

Workflow:

1. User enters a question.
2. The question is converted into an embedding using a SentenceTransformer model.
3. Pinecone performs vector similarity search to retrieve relevant documents.
4. Retrieved context is passed to the Flan-T5 model.
5. The model generates a final answer based on the context.

This approach improves answer quality compared to standalone language models.

---

## Features

- Semantic vector search using Pinecone
- SentenceTransformer embeddings
- Text generation using Flan-T5
- Top-K document retrieval
- Display of retrieved context
- Streamlit-based interactive interface
- Cached embedding model loading for performance

---

## Technologies Used

- Python
- Streamlit
- SentenceTransformers
- Pinecone Vector Database
- Hugging Face Transformers
- Flan-T5 Model

---

## Models Used

Embedding Model:
sentence-transformers/all-roberta-large-v1

Generation Model:
google/flan-t5-base

Vector Database:
Pinecone Index

---


