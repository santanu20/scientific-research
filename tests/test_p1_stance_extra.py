"""P1-11 stance extra-data loader tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestExtraStanceData:
    def test_extra_data_merged(self, tmp_path, monkeypatch):
        import _stance_svm

        extra = tmp_path / "extra.jsonl"
        extra.write_text(
            '{"text": "Our data contradict the model of Lee et al.", "label": "contrasting"}\n'
            '{"text": "We extend the framework of Diaz (2020).", "label": "extending", "field": "cs"}\n'
        )
        monkeypatch.setenv("SCIENTIFIC_RESEARCH_STANCE_DATA", str(extra))
        base = _stance_svm._load_training_data()
        assert sum(1 for d in base if d.get("field") == "extra") == 1  # default field
        assert sum(1 for d in base if d.get("field") == "cs") == 1  # explicit field

    def test_invalid_labels_skipped(self, tmp_path, monkeypatch):
        import _stance_svm

        extra = tmp_path / "extra.jsonl"
        extra.write_text(
            '{"text": "ok", "label": "supporting"}\n'
            '{"text": "bad label", "label": "background"}\n'  # SciCite intent, not stance
            "not json at all\n"
        )
        monkeypatch.setenv("SCIENTIFIC_RESEARCH_STANCE_DATA", str(extra))
        data = _stance_svm._load_training_data()
        extras = [d for d in data if d.get("field") == "extra"]
        assert len(extras) == 1

    def test_missing_path_falls_back(self, tmp_path, monkeypatch):
        import _stance_svm

        monkeypatch.setenv(
            "SCIENTIFIC_RESEARCH_STANCE_DATA", str(tmp_path / "nope.jsonl")
        )
        data = _stance_svm._load_training_data()
        assert all(d.get("field") != "extra" for d in data)

    def test_unset_env_is_default(self, monkeypatch):
        import _stance_svm

        monkeypatch.delenv("SCIENTIFIC_RESEARCH_STANCE_DATA", raising=False)
        data = _stance_svm._load_training_data()
        assert len(data) > 100  # bundled corpus intact
