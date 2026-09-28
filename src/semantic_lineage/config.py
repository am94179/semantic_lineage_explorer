"""Runtime configuration shared by future application components."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when an environment value cannot be used safely."""


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_EMBEDDING_PROVIDER = "openai"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-5.4-mini"


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration loaded from environment variables.

    API keys remain optional in Phase 0 because no external provider is called yet.
    """

    qdrant_url: str = DEFAULT_QDRANT_URL
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_api_key: str | None = None
    llm_model: str = DEFAULT_LLM_MODEL
    llm_api_key: str | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        """Load settings without reading or requiring a local `.env` file."""
        settings = cls(
            qdrant_url=os.getenv("SEMANTIC_LINEAGE_QDRANT_URL", DEFAULT_QDRANT_URL),
            embedding_provider=os.getenv(
                "SEMANTIC_LINEAGE_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER
            ),
            embedding_model=os.getenv("SEMANTIC_LINEAGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_api_key=os.getenv("SEMANTIC_LINEAGE_EMBEDDING_API_KEY"),
            llm_model=os.getenv("SEMANTIC_LINEAGE_LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_api_key=os.getenv("SEMANTIC_LINEAGE_LLM_API_KEY")
            or os.getenv("SEMANTIC_LINEAGE_EMBEDDING_API_KEY"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate only settings required to be structurally meaningful in this phase."""
        parsed_url = urlparse(self.qdrant_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ConfigurationError("SEMANTIC_LINEAGE_QDRANT_URL must be an absolute HTTP(S) URL")
        if not self.embedding_provider.strip():
            raise ConfigurationError("SEMANTIC_LINEAGE_EMBEDDING_PROVIDER cannot be empty")
        if not self.embedding_model.strip():
            raise ConfigurationError("SEMANTIC_LINEAGE_EMBEDDING_MODEL cannot be empty")
        if not self.llm_model.strip():
            raise ConfigurationError("SEMANTIC_LINEAGE_LLM_MODEL cannot be empty")
