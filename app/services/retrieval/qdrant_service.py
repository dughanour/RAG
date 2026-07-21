import uuid
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from qdrant_client.http import models
from app.config import settings
from loguru import logger

class QdrantService:
    """
    Centralized vector database service for Qdrant.
    Responsibilities:
        - Client connection management
        - Collection initialization (create if not exists)
        - Upserting document chunks with their embeddings
        - Similarity search (query by vector)
    """

    def __init__(self, embedding_dim: int):
        self.client = QdrantClient(
            url=settings.QDRANT_CLUSTER_ENDPOINT,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.embedding_dim = embedding_dim
        
        # Ensure collection exists on startup
        self.ensure_collection()

        logger.info(
            "QdrantService ready | collection='{}' | dim={}",
            self.collection_name, embedding_dim,
        )
    
    # ---- Public API ----
    def upsert_chunks(self, chunks: list[Document], embeddings: list[list[float]], filename: str, source_type: str) -> int:
        """
        Build Qdrant points from chunks + embeddings and upsert to the collection.
        Args:
            chunks: LangChain Documents (page_content + metadata).
            embeddings: Corresponding embedding vectors (same length as chunks).
            filename: Source document filename for payload.
            source_type: Category label for payload filtering.
        Returns:
            Number of points upserted.
        """
        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_payload = {
                "text": chunk.page_content,
                "source": filename,
                "source_type": source_type,
                **chunk.metadata, # Preserves page_number, total_pages, and images (base64/summaries)
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=point_payload,
                )
            )
        # Upsert in batches
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        logger.info("Upserted {} vectors to '{}'", len(points), self.collection_name)
        return len(points)
    
    def search(self, query_vector: list[float], top_k: int = 6) -> int:
        """
        Search the collection for the most similar vectors.
        Args:
            query_vector: The embedded query vector.
            top_k: Number of results to return.
        Returns:
            List of ScoredPoint objects with .score and .payload.
        """
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        
        return response.points
    
    def get_points_count(self) -> int:
        """Return the total number of vectors in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
    
    # ---- Private ----
    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        if exists:
            logger.info("Collection '{}' already exists", self.collection_name)
            return
        logger.info(
            "Creating collection '{}' (dim={}, distance=COSINE)...",
            self.collection_name, self.embedding_dim,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Collection created successfully")


    

        
        


        
    




