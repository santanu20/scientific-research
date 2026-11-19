"""P0-5 provenance manifest tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _provenance import (
    PROVENANCE_FILENAME,
    ProvenanceManifestError,
    load_manifest,
    record_stage,
)


class TestRecordStage:
    def test_appends_records_with_hashes(self, tmp_path):
        src = tmp_path / "in.json"
        src.write_text('{"a": 1}')
        out = tmp_path / "out.json"
        out.write_text('{"b": 2}')
        rec = record_stage(
            "discovery",
            params={"query": "test"},
            inputs={"corpus": src},
            outputs={"verified": out},
            results_dir=tmp_path,
        )
        assert rec["stage"] == "discovery"
        assert len(rec["inputs"]["corpus"]["sha256"]) == 64
        assert rec["outputs"]["verified"]["size_bytes"] > 0
        manifest = load_manifest(tmp_path)
        assert len(manifest) == 1
        assert manifest[0]["params"]["query"] == "test"

    def test_multiple_stages_accumulate(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("data")
        record_stage("discovery", {}, {}, {"corpus": f}, tmp_path)
        record_stage("verification", {}, {"corpus": f}, {}, tmp_path)
        manifest = load_manifest(tmp_path)
        assert [m["stage"] for m in manifest] == ["discovery", "verification"]

    def test_missing_file_recorded_not_hidden(self, tmp_path):
        rec = record_stage(
            "x", {}, {"gone": tmp_path / "nope.json"}, {}, results_dir=tmp_path
        )
        assert rec["inputs"]["gone"] == {
            "path": str(tmp_path / "nope.json"),
            "missing": True,
        }

    def test_corrupt_manifest_is_preserved_and_fails_closed(self, tmp_path):
        path = tmp_path / PROVENANCE_FILENAME
        original = "{not json"
        path.write_text(original)
        with pytest.raises(ProvenanceManifestError):
            record_stage("s", {}, {}, {}, results_dir=tmp_path)
        assert path.read_text() == original

    def test_env_snapshot_present(self, tmp_path):
        rec = record_stage("s", {}, {}, {}, results_dir=tmp_path)
        assert "python" in rec["env"]
        assert "captured_utc" in rec["env"]

    def test_load_manifest_absent_dir(self, tmp_path):
        assert load_manifest(tmp_path) == []
