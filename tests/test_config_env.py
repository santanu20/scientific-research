"""env_setting resolution: primary wins, legacy fallback warns, default applies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _config import env_setting


def test_env_setting__primary_variable_wins_over_legacy(monkeypatch, caplog):
    monkeypatch.setenv("SCIENTIFIC_RESEARCH_LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("GEOKIT_LLM_MODEL", "legacy-model")
    with caplog.at_level("WARNING", logger="scientific_research.config"):
        value = env_setting("SCIENTIFIC_RESEARCH_LLM_MODEL", legacy_name="GEOKIT_LLM_MODEL")
    assert value == "qwen3:8b"
    assert caplog.text == ""


def test_env_setting__legacy_alias_honored_with_deprecation_warning(monkeypatch, caplog):
    monkeypatch.delenv("SCIENTIFIC_RESEARCH_LLM_MODEL", raising=False)
    monkeypatch.setenv("GEOKIT_LLM_MODEL", "legacy-model")
    with caplog.at_level("WARNING", logger="scientific_research.config"):
        value = env_setting("SCIENTIFIC_RESEARCH_LLM_MODEL", legacy_name="GEOKIT_LLM_MODEL")
    assert value == "legacy-model"
    assert "GEOKIT_LLM_MODEL is deprecated" in caplog.text
    assert "SCIENTIFIC_RESEARCH_LLM_MODEL" in caplog.text


def test_env_setting__unset_returns_default(monkeypatch):
    monkeypatch.delenv("SCIENTIFIC_RESEARCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("GEOKIT_LLM_MODEL", raising=False)
    assert env_setting("SCIENTIFIC_RESEARCH_LLM_MODEL", legacy_name="GEOKIT_LLM_MODEL") == ""
    assert (
        env_setting(
            "SCIENTIFIC_RESEARCH_EMBEDDING_MODEL",
            default="ollama/nomic-embed-text",
            legacy_name="GEOKIT_EMBEDDING_MODEL",
        )
        == "ollama/nomic-embed-text"
    )
