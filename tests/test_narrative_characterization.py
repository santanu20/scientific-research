"""Phase 2: narrative characterization tests — golden brief structure.

Pins the STRUCTURE of the deliverable (not prose wording): every required
section present, citations resolve, no template-leak tokens, pooled section
consistent with meta input. Runs the real synthesize on a frozen 6-paper
corpus — deterministic, offline.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest

synth_path = Path(__file__).parent.parent / "scripts" / "synthesize.py"

FROZEN = {
    "papers": [
        {
            "doi": "10.1/t1",
            "title": "Amphibole thermobarometry of Mount A",
            "abstract": "amphibole thermobarometry yields 850 °C and 6 kbar storage conditions",
            "year": 2020,
        },
        {
            "doi": "10.1/t2",
            "title": "Amphibole thermobarometry of Mount B",
            "abstract": "amphibole thermobarometry yields 900 °C and 7 kbar storage conditions",
            "year": 2021,
        },
        {
            "doi": "10.1/t3",
            "title": "Amphibole thermobarometry of Mount C",
            "abstract": "amphibole thermobarometry yields 870 °C and 5 kbar storage conditions",
            "year": 2019,
        },
        {
            "doi": "10.1/t4",
            "title": "ML thermobarometry calibration",
            "abstract": "machine learning thermobarometry of amphibole compositions",
            "year": 2022,
        },
        {
            "doi": "10.1/t5",
            "title": "Amphibole in arc magma systems",
            "abstract": "amphibole fractionation controls arc magma chemistry",
            "year": 2018,
        },
        {
            "doi": "10.1/t6",
            "title": "Clinopyroxene barometry review",
            "abstract": "clinopyroxene-melt barometry review of methods",
            "year": 2017,
        },
    ]
}


def _parse(ab):
    out = []
    for m in re.finditer(r"(\d+) °C", ab):
        out.append((float(m.group(1)), "°C", "temperature"))
    for m in re.finditer(r"(\d+) kbar", ab):
        out.append((float(m.group(1)), "kbar", "pressure"))
    return out


@pytest.fixture(scope="module")
def brief(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("narrative")
    ver = tmp / "verified.json"
    ver.write_text(json.dumps(FROZEN))
    ext = {
        "extractions": [
            {
                "paper_id": f"doi:{p['doi']}",
                "doi": p["doi"],
                "title": p["title"],
                "abstract": p["abstract"],
                "effect_sizes": {
                    "single_measurements": (
                        [
                            {"value": v, "unit": u, "measurement": m}
                            for (v, u, m) in _parse(p["abstract"])
                        ]
                    )
                },
            }
            for p in FROZEN["papers"]
        ]
    }

    ext_file = tmp / "extracted.json"
    ext_file.write_text(json.dumps(ext))
    meta = {
        "unit_pools": [
            {
                "group": "temperature (°C)",
                "k": 3,
                "pooled_random": {
                    "effect": 873.3,
                    "ci_lower": 800.0,
                    "ci_upper": 950.0,
                    "heterogeneity": {"i_squared": 42.0},
                },
            }
        ]
    }
    meta_file = tmp / "meta.json"
    meta_file.write_text(json.dumps(meta))
    corr_file = tmp / "correlation.json"
    corr_file.write_text(
        json.dumps({"n_papers": 6, "bibliographic_coupling_edges": [{"a": 1}]})
    )
    out = tmp / "brief.md"
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(synth_path),
            "--query",
            "amphibole thermobarometry arc magma temperatures",
            "--extracted",
            str(ext_file),
            "--verified",
            str(ver),
            "--correlation",
            str(corr_file),
            "--meta",
            str(meta_file),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "SCIENTIFIC_RESEARCH_NO_SKILL_VENV": "1"},
    )
    assert r.returncode == 0, r.stderr[-500:]
    return out.read_text()


class TestBriefCharacterization:
    def test_required_sections_present(self, brief):
        for sec in (
            "## Executive summary",
            "## Methodological landscape",
            "## References",
        ):
            assert sec in brief, f"missing section: {sec}"

    def test_pooled_section_quotes_meta_values(self, brief):
        assert "873" in brief  # pooled temperature from frozen meta
        assert "k=3" in brief

    def test_all_inline_cites_resolve(self, brief):
        refs = set(re.findall(r"\[(\d+)\]", brief.split("## References")[0]))
        ref_entries = set(
            re.findall(r"^\[(\d+)\]", brief.split("## References")[1], re.M)
        )
        assert refs, "no inline citations in body"
        assert refs <= ref_entries, f"dangling cites: {refs - ref_entries}"

    def test_no_template_leak_tokens(self, brief):
        for leak in ("{", "TODO", "FIXME", "None.", "[FILL"):
            # braces appear in legitimate contexts rarely; check doubled braces
            if leak == "{":
                assert "{{" not in brief
            else:
                assert leak not in brief

    def test_no_false_geographic_claim(self, brief):
        assert "limited to 1 region" not in brief

    def test_word_floor(self, brief):
        assert len(brief.split()) >= 400  # frozen 6-paper corpus floor
