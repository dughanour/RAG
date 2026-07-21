import os
import json

from loguru import logger 
from app.services.retrieval.qdrant_service import QdrantService


from app.config import settings
from app.services.retrieval.embeddings import EmbeddingService
from app.ingestion.chunking.chunker import Chunker
from app.ingestion.loader.smart_document_parser import SmartDocumentParser


class DocumentProcessor:
    """
    Orchestrator for the document ingestion pipeline.
    Coordinates:
      1. Parsing  (Mistral OCR + VLM image descriptions)
      2. Chunking (Structure-aware markdown splitter)
      3. Local Metadata Archiving (JSON format)
      4. Embedding (Gemini / Local fallback)
      5. Storage   (Qdrant vector DB via QdrantService)
    """
    def __init__(self, processed_data_dir: str = "processed_data") -> None:
        self._processed_data_dir = processed_data_dir
        
        # Initialize internal processing services
        self._parser = SmartDocumentParser(summarize_images=True)
        self._chunker = Chunker()
        self._embedder = EmbeddingService()
        self._vector_db = QdrantService(embedding_dim=self._embedder.get_embedding_dim())

        logger.info(
            "DocumentProcessor initialized | local_archive='{}' | collection='{}'",
            processed_data_dir, "enterprise-rag",
        )

    
    def process_file(self, file_path: str, filename: str, source_type: str) -> None:
        """
        Execute the full ingestion pipeline for a single document.
        Args:
            file_path: Path to the physical document.
            filename: Human-readable name (e.g. 'EFESO_Big_Data.pdf').
            source_type: Category (e.g. 'finance', 'manuals').
        """
        logger.info("Starting processing pipeline for: {}", filename)
        try:
            # 1. Parse
            pages = self._parser.parse(file_path)
            if not pages:
                logger.warning("Parsing produced no pages for: {}", filename)
                return
            # 2. Chunk
            chunks = self._chunker.chunk(pages)
            if not chunks:
                logger.warning("Chunking produced 0 chunks for: {}", filename)
                return
            logger.info("Chunked | {} pages → {} chunks", len(pages), len(chunks))
            # 3. Archive locally
            archive_payload = {
                "filename": filename,
                "source_type": source_type,
                "chunks": [
                    {"page_content": c.page_content, "metadata": c.metadata}
                    for c in chunks
                ],
            }
            local_path = self._save_processed_locally(archive_payload, source_type, filename)
            logger.info("Local archive saved: {}", local_path)
            # 4. Embed
            logger.info("Generating embeddings for {} chunks...", len(chunks))
            chunk_texts = [c.page_content for c in chunks]
            embeddings = self._embedder.embed_texts(chunk_texts)
            # 5. Upsert to vector DB
            self._vector_db.upsert_chunks(chunks, embeddings, filename, source_type)
            logger.info("Pipeline completed for: {}", filename)

        except Exception as e:
            logger.exception("Pipeline failed for '{}': {}", filename, e)


    def _save_processed_locally(self, data: dict, source_type: str, filename: str) -> str:
        """Save parsed chunk metadata as JSON."""
        folder = os.path.join(self._processed_data_dir, source_type)
        os.makedirs(folder, exist_ok=True)
        dest_path = os.path.join(folder, f"{filename}.json")
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return dest_path

