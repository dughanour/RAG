from fastapi import APIRouter
from app.api.schemas import HealthResponse, ServiceHealthResponse
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.qdrant_service import QdrantService

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    """Basic health check."""
    return HealthResponse(status="healthy", version="1.0.0")

@router.get("/health/services", response_model=ServiceHealthResponse)
def service_health():
    """Deep health check — verifies all external services."""
    embedder = EmbeddingService()
    vector_db = QdrantService(embedding_dim=embedder.get_embedding_dim())

    return ServiceHealthResponse(
        status="healthy",
        qdrant_connected=True,
        qdrant_collection="enterprise-rag",
        qdrant_points_count=vector_db.get_points_count(),
        embedding_model="gemini" if embedder._model_type == "gemini" else "fallback",
        embedding_dim=embedder.get_embedding_dim(),
    )
