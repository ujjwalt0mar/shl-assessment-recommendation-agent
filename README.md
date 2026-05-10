# 🔍 SHL Assessment Recommendation Agent

A **Conversational RAG (Retrieval-Augmented Generation) Agent** that recommends relevant SHL assessments based on hiring requirements provided by recruiters or hiring managers.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Railway](https://img.shields.io/badge/Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app/)

**🚀 Live API:** [`https://shl-assessment-recommendation-agent-production.up.railway.app`](https://shl-assessment-recommendation-agent-production.up.railway.app)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Challenges & Future Work](#challenges--future-work)

---

## Overview

This system uses:

- **Semantic search** with FAISS for efficient retrieval
- **Sentence Transformers** for embedding generation
- **FastAPI** for a clean, scalable backend
- **Stateless conversational design** for multi-turn interactions

The agent can understand hiring requirements, ask clarifying questions for vague inputs, and return structured JSON recommendations — with support for follow-up refinement.

---

## Features

- 🤖 Conversational multi-turn recommendation system
- 🗄️ SHL assessment retrieval via semantic similarity (FAISS)
- 💬 Clarification questions for ambiguous queries
- 🔁 Recommendation refinement through follow-up conversation
- 📦 Structured JSON API responses
- ♻️ Fully stateless — context managed client-side
- ✅ Health check endpoint
- 🛡️ Safety-aware conversational handling

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **FastAPI** | Backend API framework |
| **FAISS** | Vector similarity search |
| **Sentence Transformers** | Text embedding generation |
| **Python** | Core programming language |
| **Railway** | Cloud deployment platform |
| **Uvicorn** | ASGI server |

---

## Project Structure

```
.
├── app/
│   ├── main.py         # Entry point
│   ├── routes.py       # API route definitions
│   ├── retriever.py    # FAISS retrieval logic
│   ├── agent.py        # Conversational agent
│   ├── prompts.py      # Prompt templates
│   ├── models.py       # Pydantic models
│   ├── config.py       # Configuration
│   ├── utils.py        # Utility functions
│   └── scraper.py      # Data collection
│
├── data/
│   ├── assessments.json
│   └── faiss_index/
│       ├── index.faiss
│       └── metadata.json
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## API Reference

### `GET /health`

Checks whether the API is running.

```bash
curl https://shl-assessment-recommendation-agent-production.up.railway.app/health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

### `POST /chat`

Main conversational endpoint for assessment recommendations.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Need aptitude and reasoning assessments for graduate hiring."
    }
  ]
}
```

**Example cURL:**
```bash
curl -X POST \
  'https://shl-assessment-recommendation-agent-production.up.railway.app/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hiring Java backend developers with communication skills."
      }
    ]
  }'
```

**Example Response:**
```json
{
  "reply": "Here are recommended SHL assessments for Java backend developer hiring.",
  "recommendations": [
    {
      "name": "Core Java Assessment",
      "url": "https://www.shl.com/",
      "reason": "Evaluates Java programming fundamentals."
    }
  ],
  "end_of_conversation": false
}
```

---

### Stateless Multi-Turn Conversation

The API is fully stateless. Pass the full conversation history in every request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Need assessments for software engineers."
    },
    {
      "role": "assistant",
      "content": "Which programming language or role specialization are you hiring for?"
    },
    {
      "role": "user",
      "content": "Python backend developers."
    }
  ]
}
```

---

## How It Works

```
User Query
    │
    ▼
Embedding Generation (sentence-transformers/all-MiniLM-L6-v2)
    │
    ▼
FAISS Semantic Search → Top-K Relevant Assessments
    │
    ▼
Conversational Agent (clarify / recommend / refine)
    │
    ▼
Structured JSON Response
```

1. **Data Collection** — SHL assessment metadata is stored in `data/assessments.json`
2. **Embedding Generation** — Descriptions are embedded using `all-MiniLM-L6-v2`
3. **Vector Indexing** — Embeddings are indexed with FAISS for fast retrieval
4. **Conversational Recommendation** — The agent understands intent, retrieves assessments, asks follow-up questions, and returns structured recommendations

---

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/ujjwalt0mar/shl-assessment-recommendation-agent.git
cd shl-assessment-recommendation-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

---

## Deployment

The application is deployed on **Railway**.

Key deployment configurations:
- Python version pinned to `3.11.9` via `runtime.txt`
- FAISS compatibility handling for Railway's environment
- Memory optimization for free-tier constraints
- Custom start command configuration

---

## Challenges & Future Work

### Challenges Faced

- Render free tier memory limitations
- FAISS dependency compatibility across environments
- Python 3.14 incompatibility issues
- Railway deployment configuration tuning
- Semantic retrieval quality tuning

### Planned Improvements

- [ ] Hybrid retrieval (semantic + keyword search)
- [ ] Better reranking strategies
- [ ] Assessment comparison support
- [ ] Improved conversational memory
- [ ] Streaming responses
- [ ] Caching optimization

---

## Author

**Ujjwal Tomar**  
B.Tech — Artificial Intelligence & Data Science

---

## License

This project is created for educational and assignment purposes.