import logfire
import os
import uuid
import json

from loguru import logger 
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embeddings import EmbeddingService
from app.ingestion.chunking.chunker import Chunker
from app.ingestion.loader.smart_document_parser import SmartDocumentParser


class DocumentProcessor:
    """
    Orchestrator for the document ingestion pipeline.
    
    Coordinates:
      1. Parsing (Mistral OCR + VLM image descriptions)
      2. Chunking (Structure-aware markdown splitter)
      3. Local Metadata Archiving (JSON format)
      4. Vector Embeddings (Gemini / Local fallback)
      5. DB Storage (Upserting vectors & payload to Qdrant)
    """
    def __init__(self, processed_data_dir: str = "processed_data") -> None:
        self._processed_data_dir = processed_data_dir
        
        # Initialize internal processing services
        self._parser = SmartDocumentParser(summarize_images=True)
        self._chunker = Chunker()
        self._embedder = EmbeddingService()
        
        # Initialize Qdrant Client
        self._qdrant = QdrantClient(
            url=settings.QDRANT_CLUSTER_ENDPOINT,
            api_key=settings.QDRANT_API_KEY,
        )
        
        logger.info(
            "DocumentProcessor initialized | local_archive='{}' | collection='{}'",
            processed_data_dir,
            settings.QDRANT_COLLECTION_NAME
        )
    
    def process_file(self, file_path: str, filename: str, source_type: str) -> None:
        """
        Execute the ingestion pipeline for a single document.
        
        Args:
            file_path: Absolute or relative path to the physical document.
            filename: Human-readable identifier/filename (e.g. 'EFESO_Big_Data.pdf').
            source_type: Subfolder categorize path (e.g. 'finance', 'manuals').
        """
        logger.info("Starting processing pipeline for: {}", filename)
        
        try:
            # 1. Parse document using Mistral OCR + Image VLM Summaries
            pages = self._parser.parse(file_path)
            if not pages:
                logger.warning("Parsing produced no pages for file: {}", filename)
                return
            # 2. Split page markdown contents by header hierarchies
            chunks = self._chunker.chunk(pages)
            if not chunks:
                logger.warning("Chunking produced 0 chunks for file: {}", filename)
                return
            logger.info("Document chunked | pages={} -> chunks={}", len(pages), len(chunks))
            # 3. Archive parsing metadata locally as JSON
            # Note: We convert Document objects to dictionary representations
            # to prevent JSON serialization errors.
            archive_payload = {
                "filename": filename,
                "source_type": source_type,
                "chunks": [
                    {
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunks
                ]
            }
            local_path = self._save_processed_locally(archive_payload, source_type, filename)
            logger.info("Local metadata archive created at: {}", local_path)
            # 4. Generate Vector Embeddings (Gemini 3072 / Local 768 fallback)
            logger.info("Generating embeddings for {} chunks...", len(chunks))
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = self._embedder.embed_texts(chunk_texts)
            # 5. Build Qdrant Point structures and upsert
            # Note: We merge chunk.metadata (including image base64, VLM summaries, 
            # and page numbers) directly into the vector database payload.
            points = []
            for chunk, vector in zip(chunks, embeddings):
                point_payload = {
                    "text": chunk.page_content,
                    "source": filename,
                    "source_type": source_type,
                    **chunk.metadata  # Preserves page_number, total_pages, and images base64/summaries
                }
                
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=point_payload
                    )
                )
            logger.info("Upserting {} vectors to Qdrant...", len(points))
            self._qdrant.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points,
            )
            logger.info("Ingestion pipeline completed successfully for: {}", filename)
        except Exception as e:
            logger.exception("Ingestion pipeline failed for file '{}': {}", filename, e)
    
    def _save_processed_locally(self, data: dict, source_type: str, filename: str) -> str:
        """Save structured parsed document metadata to the local directory."""
        folder = os.path.join(self._processed_data_dir, source_type)
        os.makedirs(folder, exist_ok=True)
        
        dest_path = os.path.join(folder, f"{filename}.json")
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return dest_path

