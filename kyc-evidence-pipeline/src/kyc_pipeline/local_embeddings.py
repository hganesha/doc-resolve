"""Optional local dense retrieval. Only model weights are downloaded; text stays local."""
from __future__ import annotations

from .extraction import ProviderError

DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class LocalEmbeddingProvider:
    def __init__(self, model: str = DEFAULT_LOCAL_EMBEDDING_MODEL):
        self.embedding_model = model
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ProviderError("Local embeddings require pip install -e '.[local-embeddings]'") from None
        try:
            # Bound CPU consumption; first use downloads model weights unless already cached.
            self._model = TextEmbedding(model_name=model, threads=2)
        except Exception as exc:
            raise ProviderError(f"Could not load local embedding model ({type(exc).__name__}); check model name, cache and network") from None

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return [x.tolist() for x in self._model.embed(texts, batch_size=4)]
        except Exception as exc:
            raise ProviderError(f"Local document embedding failed ({type(exc).__name__})") from None

    def embed_query(self, text: str) -> list[float]:
        try:
            return next(iter(self._model.query_embed([text]))).tolist()
        except Exception as exc:
            raise ProviderError(f"Local query embedding failed ({type(exc).__name__})") from None
