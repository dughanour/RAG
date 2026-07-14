import os
import sys
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.ingestion.processor import DocumentProcessor
from app.services.retrieval.embeddings import EmbeddingService

def setup_qdrant_collection(client: QdrantClient, vector_size: int):
    """Ensure the target collection exists in Qdrant with the correct configuration."""
    collection_name = settings.QDRANT_COLLECTION_NAME
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(col.name == collection_name for col in collections)
    
    if exists:
        logger.info(f"Qdrant collection '{collection_name}' already exists.")
        return

    logger.info(f"Creating Qdrant collection '{collection_name}' (dim={vector_size})...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )
    logger.info("Collection created successfully.")

def run_ingestion():
    """Runs the document parser, chunker, and database upload."""
    file_path = os.path.join("data", "DEFR PEFT.pdf")
    
    if not os.path.exists(file_path):
        logger.error(f"Test file not found at: {file_path}")
        return

    try:
        logger.info("Initializing DocumentProcessor...")
        processor = DocumentProcessor()
        
        # Determine the active embedding dimension from the processor's embedder
        embedding_dim = processor._embedder.get_embedding_dim()
        
        # Ensure Qdrant is configured
        setup_qdrant_collection(processor._qdrant, embedding_dim)

        # Run the full pipeline
        processor.process_file(
            file_path=file_path,
            filename=os.path.basename(file_path),
            source_type="Pdf"
        )

        # Verify the upload
        collection_info = processor._qdrant.get_collection(settings.QDRANT_COLLECTION_NAME)
        logger.info(f"Verification: Qdrant collection contains {collection_info.points_count} total vectors.")

        local_archive = os.path.join("processed_data", "inflation_management", f"{os.path.basename(file_path)}.json")
        if os.path.exists(local_archive):
            logger.info(f"Verification: Local metadata archive created at '{local_archive}'")
        else:
            logger.error("Verification failed: Local metadata archive was not found.")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

def run_retrieval():
    """Takes a console query, embeds it, and prints the top 3 Qdrant results."""
    query = input("\nEnter your search query: ").strip()
    if not query:
        logger.warning("Empty query. Aborting search.")
        return

    try:
        logger.info("Initializing EmbeddingService and QdrantClient...")
        embedder = EmbeddingService()
        qdrant = QdrantClient(
            url=settings.QDRANT_CLUSTER_ENDPOINT,
            api_key=settings.QDRANT_API_KEY,
        )

        logger.info("Generating query embedding...")
        query_vector = embedder.embed_query(query)

        logger.info("Searching Qdrant for top 3 matches...")
        response  = qdrant.query_points(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=3
        )

        print("\n" + "="*80)
        print(f"RETRIEVED RESULTS FOR: '{query}'")
        print("="*80)

        for i, hit in enumerate(response .points):
            payload = hit.payload
            score = hit.score
            page = payload.get("page_number", "?")
            source = payload.get("source", "unknown")
            text = payload.get("text", "")

            print(f"\n[Result {i+1}] Score: {score:.4f} | Page: {page} | Source: {source}")
            print(f"[Content]:\n{text}")
            print("-" * 80)

        print("\n" + "="*80)

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")


def main():
    while True:
        print("\n" + "="*30 + " MENU " + "="*30)
        print("1 - Search (Retrieve top 3 relevant chunks)")
        print("2 - Ingest (Run ingestion pipeline on target PDF)")
        print("3 - Exit")
        print("="*66)
        
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            run_retrieval()
        elif choice == "2":
            run_ingestion()
        elif choice == "3":
            logger.info("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
