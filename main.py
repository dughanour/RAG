from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, ingestion, health

app = FastAPI(
    title="Enterprise RAG API",
    description="Agentic RAG system with Mistral OCR, Gemini Embeddings, and Groq LLM",
    version="1.0.0",
)

# CORS — allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(ingestion.router, prefix="/api", tags=["Ingestion"])
app.include_router(health.router, prefix="/api", tags=["Health"])
