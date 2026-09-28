"""Embedding-provider interface and implementations for semantic catalog retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

from openai import OpenAI

from semantic_lineage.config import Settings


class EmbeddingProviderError(ValueError):
    """Raised when an embedding provider cannot create valid vectors."""


class EmbeddingProvider(Protocol):
    """Small provider boundary so retrieval is not coupled to one embedding API."""

    @property
    def provider_name(self) -> str:
        """Return a stable identifier for evaluation provenance."""

    @property
    def model_name(self) -> str:
        """Return the embedding model or deterministic algorithm identifier."""

    @property
    def dimensions(self) -> int:
        """Return the fixed dimension of vectors produced by the provider."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed catalog documents in input order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one user query."""


class OpenAIEmbeddingProvider:
    """Hosted embedding provider used by the normal runtime path."""

    def __init__(self, model: str, api_key: str | None, dimensions: int = 1536) -> None:
        if not api_key:
            raise EmbeddingProviderError(
                "SEMANTIC_LINEAGE_EMBEDDING_API_KEY is required for the OpenAI embedding provider"
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [list(item.embedding) for item in response.data]
        self._validate_vectors(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0]

    def _validate_vectors(self, vectors: list[list[float]]) -> None:
        if any(len(vector) != self.dimensions for vector in vectors):
            raise EmbeddingProviderError(
                f"embedding model '{self._model}' did not return {self.dimensions}-dimension vectors"
            )


class DeterministicHashEmbeddingProvider:
    """Dependency-free, deterministic embedder for tests and local demonstrations only.

    It is intentionally not a semantic model and must not be used to assess retrieval quality.
    """

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 2:
            raise EmbeddingProviderError("dimensions must be at least 2")
        self._dimensions = dimensions

    @property
    def provider_name(self) -> str:
        return "deterministic_hash"

    @property
    def model_name(self) -> str:
        return "sha256_token_hash"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        token_counts = Counter(re.findall(r"[a-z0-9_]+", text.lower()))
        vector = [0.0] * self.dimensions
        for token, count in token_counts.items():
            token_hash = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(token_hash[:8], "big") % self.dimensions
            sign = 1.0 if token_hash[8] % 2 == 0 else -1.0
            vector[index] += sign * count
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured production embedding provider."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings.embedding_model, settings.embedding_api_key)
    raise EmbeddingProviderError(
        f"unsupported embedding provider '{settings.embedding_provider}'. Supported provider: openai"
    )
