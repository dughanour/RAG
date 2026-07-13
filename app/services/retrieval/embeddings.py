import logfire
import time
import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from app.config import settings

BATCH_SIZE = 50
_GEMINI_DIM= 3072
_FALLBACK_DIM = 768 # all-mpnet_base-v2

_active_model = None
_model_type: str | None = None

def _probe_gemini():
    """ Try one embed call to verify connectivity """
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            api_key=settings.GEMINI_API_KEY
        )
        model.embed_query("test")
        logfire.info("Gemini connectivity check passed")
        return model
    except Exception as e:
        logfire.warning(f"Gemini connectivity check failed: {e}")
        return None

def _load_fallback():
    """ Load local sentence-transformers model """
    try:
        model = SentenceTransformer(
            "all-mpnet-base-v2"
        )
        logfire.info("Fallback model (all-mpnet-base-v2) loaded successfully")
        return model
        
    except Exception as e:
        logfire.warning(f"Fallback model loading failed: {e}")
        return None

def _init():
    """ Init embedding model """
    global _active_model, _model_type

    if _active_model is not None:
        return
    
    gemini = _probe_gemini()
    
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


def get_embedding_dim() -> int:
    """ Return the dimension of the embedding model """
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM

def _embed_batch(batch: list[str]) -> list[list[float]]:
    """ Embed a batch of text """
    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err =  str(e).lower()
                is_rate_limit = any(x in err for x in ("429","rate","quota","resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Gemini rate limit hit - retrying in {wait}s"
                        f"(attmept {attempt+1}/4)."
                    )
                    time.sleep(wait)
                
                else:
                    logfire.error(f"Failed to embed with Gemini: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attempts.")
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()
            

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        with logfire.span("Embed query", model=_model_type):
            return _active_model.embed_query(query)
    else:
        return _active_model.encode([query])[0].tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings




