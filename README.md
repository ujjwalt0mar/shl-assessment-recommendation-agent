# Stateless Conversational SHL Recommendation Agent

Production-oriented FastAPI backend that helps recruiters discover SHL assessments through a stateless conversational API.

## Features

- Stateless chat endpoint (`POST /chat`) with strict response schema.
- Clarification-first conversation strategy before recommendations.
- Hybrid retrieval:
  - semantic similarity (FAISS + embeddings)
  - keyword overlap
  - combined score: `0.7 * semantic + 0.3 * keyword`
- Prompt-injection resistance and out-of-domain refusal.
- Comparison support (example: OPQ vs GSA/GAS).
- Scraper pipeline (requests + BeautifulSoup) to ingest catalog data.
- Startup loading of embedding model and FAISS index.
- Typed request/response models with Pydantic.
- Tests for health, clarification, recommendation, refusal, and comparison flows.

## Project Structure

```text
project/
|
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |-- routes.py
|   |-- agent.py
|   |-- retriever.py
|   |-- prompts.py
|   |-- scraper.py
|   |-- models.py
|   |-- utils.py
|   `-- config.py
|
|-- data/
|   |-- assessments.json
|   `-- faiss_index/
|
|-- tests/
|   `-- test_api.py
|
|-- requirements.txt
|-- README.md
`-- .env.example
```

## Requirements

- Python 3.11+
- pip

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

### `POST /chat`

Request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer"
    }
  ]
}
```

Clarification response example:

```json
{
  "reply": "What seniority level is this role (entry, junior, mid, senior, or executive)?",
  "recommendations": [],
  "end_of_conversation": false
}
```

Recommendation response example:

```json
{
  "reply": "Here are SHL recommendations for Java Developer Role with technical, personality coverage.",
  "recommendations": [
    {
      "name": "Verify Interactive - Java",
      "url": "https://www.shl.com/solutions/products/product-catalog/verify-interactive-java/",
      "test_type": "Technical"
    }
  ],
  "end_of_conversation": false
}
```

## Stateless Design

- No session storage.
- No chat-memory persistence in backend/database.
- Client sends full conversation history on every `POST /chat` request.

## Retrieval Design

For each assessment:

1. Build an embedding vector from catalog text.
2. Retrieve semantic similarity via FAISS.
3. Compute keyword overlap score.
4. Rank by:

```text
final_score = 0.7 * semantic_similarity + 0.3 * keyword_score
```

Top 1-10 assessments are returned.

## Scraper Pipeline

Run catalog scraping:

```bash
python -m app.scraper --url "https://www.shl.com/solutions/products/product-catalog/" --output data/assessments.json
```

Extracted fields:

- assessment name
- URL
- description
- skills
- duration
- test type

## Behavior Rules Enforced

- Clarifies role/seniority/test focus when missing.
- Refines recommendations if user updates requirements (for example adding personality tests).
- Supports comparison requests for catalog-backed assessments.
- Refuses legal advice, unrelated topics, and prompt-injection attempts.
- Never returns assessments outside the loaded catalog.

## Tests

```bash
pytest -q
```

## Deployment

### Render

1. Create a new Web Service from your repo.
2. Runtime: Python 3.11+
3. Build command:

```bash
pip install -r requirements.txt
```

4. Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

5. Add environment variables from `.env.example` as needed.

### Railway

1. Create a new project and connect repository.
2. Railway auto-detects Python.
3. Set start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

4. Add `.env` variables in Railway dashboard.

## cURL Examples

Health:

```bash
curl -X GET "http://localhost:8000/health"
```

Chat:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a Java developer"},
      {"role": "user", "content": "Mid-level"},
      {"role": "user", "content": "Need technical and personality tests"}
    ]
  }'
```

## Notes

- The app attempts to load `sentence-transformers/all-MiniLM-L6-v2`.
- If model loading fails (network/restricted environment), a deterministic hashing encoder fallback is used so the API remains functional.
- For best retrieval quality in production, allow model download during first startup.
