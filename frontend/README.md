# teacher frontend

The `frontend` directory contains a Streamlit app for testing the backend `/ask` API.

---

## Features

- Send questions to the backend API
- Run health checks
- Inspect returned sources
- View raw JSON responses

---

## Run Locally

docker compose -f docker/docker-compose.local.yml up --build

Open:
http://localhost:8501

---

## Environment Variables

API_BASE_URL=http://localhost:8000
