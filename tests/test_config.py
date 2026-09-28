from __future__ import annotations

import pytest

from semantic_lineage.config import ConfigurationError, Settings


def test_settings_loads_defaults_when_environment_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMANTIC_LINEAGE_QDRANT_URL", raising=False)
    monkeypatch.delenv("SEMANTIC_LINEAGE_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIC_LINEAGE_EMBEDDING_MODEL", raising=False)

    settings = Settings.from_environment()

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.embedding_provider == "openai"


def test_settings_rejects_non_http_qdrant_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_LINEAGE_QDRANT_URL", "localhost:6333")

    with pytest.raises(ConfigurationError, match="absolute HTTP"):
        Settings.from_environment()
