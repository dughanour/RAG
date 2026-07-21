from loguru import logger
from app.agents.state import AgentState
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.qdrant_service import QdrantService
from app.services.retrieval.reranking_service import rerank_documents


# Initialize services once at module level (reused across all calls)
_embedder = EmbeddingService()
_vector_db = QdrantService(embedding_dim=_embedder.get_embedding_dim())


def retriever_node(state: AgentState) -> dict:
    """
    Performs vector search + semantic re-ranking.
    Flow:
      1. Embed the planner's search query into a vector
      2. Retrieve top-k candidates from Qdrant
      3. Re-rank with FlashRank cross-encoder
      4. Return top 3 most relevant chunks
    """

    query = state["current_query"]
    logger.info("Retriever searching for: '{}'", query[:80])
    # 1. Embed the query
    query_vector = _embedder.embed_query(query)
    # 2. Vector search — retrieve broad candidates
    raw_results = _vector_db.search(query_vector, top_k=6)
    logger.info("Retrieved {} candidates from Qdrant", len(raw_results))
    # 3. Extract text content from scored points
    doc_texts = [hit.payload.get("text", "") for hit in raw_results]
    # 4. Re-rank for precision
    reranked_docs = rerank_documents(query, doc_texts, top_n=3)
    logger.info("Re-ranking complete — kept top {} documents", len(reranked_docs))
    return {
        "documents": reranked_docs,
        "status": "Technical context retrieved and re-ranked",
        "plan": state["plan"] + ["Context Retrieved"],
    }

      
        


    
  