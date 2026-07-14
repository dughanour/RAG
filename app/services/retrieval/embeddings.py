import time
from loguru import logger
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from app.config import settings




class EmbeddingService:
    """
    Embedding service with Gemini primary and local fallback.
    - Primary:  Gemini embedding-2-preview (3072 dims)
    - Fallback: all-mpnet-base-v2 via sentence-transformers (768 dims)
    Usage:
        embedder = EmbeddingService()
        vector = embedder.embed_query("some text")
        vectors = embedder.embed_texts(["text1", "text2"])
        dim = embedder.get_embedding_dim()
    """

    _GEMINI_MODEL = "models/gemini-embedding-2-preview"
    _FALLBACK_MODEL = "all-mpnet-base-v2"
    _GEMINI_DIM = 3072
    _FALLBACK_DIM = 768
    _BATCH_SIZE = 50
    _MAX_RETRIES = 4

    def __init__(self):
        self._model = None
        self._model_type: str | None = None
        self._initialize()
    
    # ---- Public API ----
    def get_embedding_dim(self) -> int:
        """Return the dimension of the active embedding model."""
        return self._GEMINI_DIM if self._model_type == "gemini" else self._FALLBACK_DIM

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        if self._model_type == "gemini":
            return self._model.embed_query(query)
        return self._model.encode([query])[0].tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            all_embeddings.extend(self._embed_batch(batch))
            logger.debug("Embedded batch {}-{} / {}", i, i + len(batch), len(texts))
        return all_embeddings
    
    # ---- Private: Initialization ----
    def _initialize(self) -> None:
        """Try Gemini first, fall back to local model."""
        gemini = self._probe_gemini()
        if gemini:
            self._model = gemini
            self._model_type = "gemini"
        else:
            self._model = self._load_fallback()
            self._model_type = "fallback"
        logger.info(
            "EmbeddingService ready | model={} | dim={}",
            self._model_type, self.get_embedding_dim(),
        )
    
    def _probe_gemini(self):
        """Try a single embed call to verify Gemini connectivity."""
        try:
            model = GoogleGenerativeAIEmbeddings(
                model=self._GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
            )
            model.embed_query("test")
            logger.info("Gemini connectivity check passed")
            return model
        except Exception as e:
            logger.warning("Gemini connectivity check failed: {}", e)
            return None
    
    def _load_fallback(self):
        """Load local sentence-transformers model."""
        logger.info("Loading fallback model: {}", self._FALLBACK_MODEL)
        return SentenceTransformer(self._FALLBACK_MODEL)
    
    # ---- Private: Batch Embedding ----

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed a batch of texts with retry logic for Gemini rate limits."""
        if self._model_type == "gemini":
            return self._embed_batch_gemini(batch)
        return self._model.encode(batch, show_progress_bar=False).tolist()

    def _embed_batch_gemini(self, batch: list[str]) -> list[list[float]]:
        """Gemini embedding with exponential backoff on rate limits."""
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(
                    x in err for x in ("429", "rate", "quota", "resource_exhausted")
                )
                if is_rate_limit and attempt < self._MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate limit hit — retrying in {}s (attempt {}/{})",
                        wait, attempt + 1, self._MAX_RETRIES,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Gemini embedding failed: {}", e)
                    raise
        raise RuntimeError(
            f"Gemini rate limit persisted after {self._MAX_RETRIES} attempts"
        )
