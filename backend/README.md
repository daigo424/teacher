# teacher backend

The `backend` directory contains the FastAPI-based RAG API, ingestion pipeline, and database-related logic for the `teacher` project.

This backend demonstrates a production-like Retrieval-Augmented Generation (RAG) workflow using Docker, PostgreSQL, Atlas, and OpenAI APIs.

---

## Tech Stack

- Python
- FastAPI
- PostgreSQL (Vector Store)
- OpenAI API (Embeddings & LLM)
- Docker / Docker Compose
- Atlas (Schema Management)

---

## Overview

This project provides a full pipeline for:

1. scraping Wikipedia pages into Markdown
2. chunking and embedding documents
3. storing them in a vector database
4. retrieving relevant context
5. generating answers via LLM

---

## Architecture

Wikipedia URL
     ↓
[wikipedia_to_markdown]
     ↓
Markdown files (files/dataset)
     ↓
[ingest]
     ↓
PostgreSQL (vector DB)
     ↓
[api (/ask)]
     ↓
OpenAI API → Answer

---

## Applications

### wikipedia_to_markdown
- Scrapes Wikipedia pages
- Converts content into clean Markdown
- Saves files under `files/dataset/`

### ingest
- Splits Markdown into chunks
- Generates embeddings
- Stores vectors into PostgreSQL

### api
- Provides `/ask` endpoint
- Retrieves relevant chunks
- Calls OpenAI API to generate answers

---

## Key Features

- Wikipedia-based dataset generation
- Chunk-based semantic search
- PostgreSQL-backed vector storage
- End-to-end RAG pipeline
- Docker-based reproducible environment

---

## Design Highlights

### Chunking Strategy
Documents are split into manageable chunks before embedding to improve retrieval accuracy and avoid context overflow.

### Meta Question Classification
Questions are classified into:
- META (about the system itself)
- CONTENT (about stored knowledge)

This prevents irrelevant retrieval and improves answer quality.

---

## Setup Notes

### Environment Setup

Copy `.env.example` and set your environment variables:

cp .env.example .env

---

### Database Schema

You can apply the database schema defined under `/db/**/*.hcl` using:

make atlas-apply

---

### Running the Application

make build  
make up  

Swagger UI will be available at:

http://localhost:8000/docs

---

### API Testing

You can test the `/ask` endpoint directly from Swagger UI.

This allows you to:
- send questions
- verify retrieval behavior
- confirm LLM-generated answers

---

### Environment Variables

You must set your OpenAI API key in `.env`:

OPENAI_API_KEY=your_api_key_here

Without this:
- ingestion will fail (no embeddings)
- `/ask` will not work

---

## Example Workflow

1. Scrape data:

make wikipedia_to_markdown URL="https://en.wikipedia.org/wiki/Artificial_intelligence"

2. Ingest:

make ingest FILEPATH=files/dataset/artificial_intelligence.md

3. Ask:

POST /ask
{
  "question": "What is artificial intelligence?"
}

---

## Summary

This backend demonstrates how to build a practical RAG system:

- collect knowledge
- process documents
- store embeddings
- retrieve relevant context
- generate answers using LLM
