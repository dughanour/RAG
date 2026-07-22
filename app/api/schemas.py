from pydantic import BaseModel

# ---- Chat ----
class ChatRequest(BaseModel):
    message: str
    session_id : str

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    planner_decision: str
    status: str
    plan: list[str]
    retrieved_chunks: int 

class NewSessionResponse(BaseModel):
    session_id: str

# ---- Ingestion ----
class IngestionResponse(BaseModel):
    filename: str
    source_type: str
    chunks_created: int
    vectors_upserted: int
    status: str

# ---- Health ----
class HealthResponse(BaseModel):
    status: str
    version: str

class ServiceHealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    qdrant_collection: str
    qdrant_points_count: int
    embedding_model: str
    embedding_dim: int