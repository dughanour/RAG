import os
from loguru import logger

from app.ingestion.processor import DocumentProcessor
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.qdrant_service import QdrantService


def run_ingestion():
    """Runs the full ingestion pipeline on a target PDF."""
    file_path = os.path.join("data", "DEFR PEFT.pdf")

    if not os.path.exists(file_path):
        logger.error(f"Test file not found at: {file_path}")
        return

    try:
        processor = DocumentProcessor()

        processor.process_file(
            file_path=file_path,
            filename=os.path.basename(file_path),
            source_type="Pdf",
        )

        # Verify
        count = processor._vector_db.get_points_count()
        logger.info(f"Verification: Qdrant contains {count} total vectors.")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")


def run_retrieval():
    """Takes a console query and prints the top 3 Qdrant results."""
    query = input("\nEnter your search query: ").strip()
    if not query:
        logger.warning("Empty query. Aborting search.")
        return

    try:
        embedder = EmbeddingService()
        vector_db = QdrantService(embedding_dim=embedder.get_embedding_dim())

        query_vector = embedder.embed_query(query)
        results = vector_db.search(query_vector, top_k=3)

        print("\n" + "="*80)
        print(f"RETRIEVED RESULTS FOR: '{query}'")
        print("="*80)

        for i, hit in enumerate(results):
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


def run_agent():
    """Runs a multi-turn chat session using the LangGraph Agentic RAG workflow."""
    import uuid
    from app.agents.graph import app as agent_app

    thread_id = str(uuid.uuid4())
    logger.info("Starting Agent Chat Session | thread_id='{}'", thread_id)

    print("\n" + "="*80)
    print("AGENTIC RAG MULTI-TURN CHAT (Type 'exit' or 'back' to return to menu)")
    print("="*80)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("exit", "back"):
            logger.info("Ending Agent session...")
            break

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [{"role": "user", "content": user_input}]}

        try:
            result = agent_app.invoke(inputs, config=config)

            print("\n" + "-"*35 + " AGENT EXECUTION TRACE " + "-"*35)
            print(f"Planner Decision : {result.get('current_query', 'N/A')}")
            print(f"Pipeline Status  : {result.get('status', 'N/A')}")
            print(f"Plan Steps       : {' -> '.join(result.get('plan', []))}")
            print(f"Retrieved Chunks : {len(result.get('documents', []))}")
            print("-" * 93)
            print(f"\nAssistant:\n{result.get('final_answer', '')}")
            print("=" * 93)

        except Exception as e:
            logger.error("Agent execution failed: {}", e)


def run_reranking_test():
    """Retrieves 6 candidates from Qdrant, reranks with FlashRank (6 -> 3), and displays Selected vs Excluded chunks."""
    from app.services.retrieval.reranking_service import rerank_documents

    query = input("\nEnter your search query for Reranking test: ").strip()
    if not query:
        logger.warning("Empty query. Aborting search.")
        return

    try:
        embedder = EmbeddingService()
        vector_db = QdrantService(embedding_dim=embedder.get_embedding_dim())

        logger.info("Fetching top 6 candidates from Qdrant vector DB...")
        query_vector = embedder.embed_query(query)
        candidates = vector_db.search(query_vector, top_k=6)

        if not candidates:
            logger.warning("No candidates retrieved from Qdrant.")
            return

        doc_texts = [hit.payload.get("text", "") for hit in candidates]

        logger.info("Applying FlashRank Cross-Encoder reranking (6 -> 3)...")
        selected_texts = rerank_documents(query, doc_texts, top_n=3)

        print("\n" + "="*80)
        print(f"FLASHRANK RERANKING ANALYSIS FOR: '{query}'")
        print("="*80)

        print("\n✅ SELECTED BY FLASHRANK (TOP 3 MOST RELEVANT):")
        print("="*80)
        for i, text in enumerate(selected_texts):
            orig_pos = doc_texts.index(text) + 1 if text in doc_texts else "?"
            print(f"\n[Rank #{i+1} | Original Qdrant Position: Candidate #{orig_pos}]")
            print(f"{text[:400]}...")
            print("-" * 60)

        print("\n❌ EXCLUDED BY FLASHRANK (REJECTED CHUNKS):")
        print("="*80)
        excluded_texts = [t for t in doc_texts if t not in selected_texts]
        for i, text in enumerate(excluded_texts):
            orig_pos = doc_texts.index(text) + 1
            print(f"\n[Rejected #{i+1} | Original Qdrant Position: Candidate #{orig_pos}]")
            print(f"{text[:400]}...")
            print("-" * 60)

        print("\n" + "="*80)

    except Exception as e:
        logger.error(f"Reranking test failed: {e}")


def main():
    while True:
        print("\n" + "="*30 + " MENU " + "="*30)
        print("1 - Search (Retrieve top 3 relevant chunks directly)")
        print("2 - Ingest (Run ingestion pipeline on target PDF)")
        print("3 - Agent Chat (Run Multi-Turn Agentic RAG Graph)")
        print("4 - FlashRank Reranking Test (Retrieve 6 -> Rerank 3 -> Show Selected vs Excluded)")
        print("5 - Exit")
        print("="*66)

        choice = input("Enter choice (1, 2, 3, 4, or 5): ").strip()

        if choice == "1":
            run_retrieval()
        elif choice == "2":
            run_ingestion()
        elif choice == "3":
            run_agent()
        elif choice == "4":
            run_reranking_test()
        elif choice == "5":
            logger.info("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()


