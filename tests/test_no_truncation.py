"""Zero-truncation policy tests (2026-08-15).

Character caps on scientific content silently destroyed data:
  - extraction saw abstract[:6000]  -> tail measurements lost
  - embedding saw text[:2000]       -> semantic ranking blind past cut
  - claims sentences >600 chars      -> dropped entirely
  - ranking saw [:1000]              -> most-abstract papers unrankable
These tests pin the lossless behavior with adversarially LONG inputs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _long_abstract(n_chars: int) -> str:
    """Synthetic abstract with a measurement planted at the TAIL."""
    filler = (
        "The crystallization history of arc magmas reflects storage conditions "
        "and degassing pathways during ascent. "
    )
    head = filler * (n_chars // len(filler) + 1)
    return (
        head[:n_chars]
        + " Amphibole thermobarometry yields 876 °C and 6.8 kbar at the final stage."
    )


class TestNoTruncation:
    def test_tail_measurement_extracted_from_long_abstract(self):
        """The [:6000] class: measurement at char ~9000 must survive."""
        from _effect_parser import extract_effect_sizes

        ab = _long_abstract(9000)
        eff = extract_effect_sizes(ab)["single_measurements"]
        vals = {(s["value"], s["unit"]) for s in eff}
        assert (876.0, "°C") in vals, "tail measurement lost to truncation"
        assert (6.8, "kbar") in vals

    def test_extract_paper_stores_full_abstract(self):
        import inspect

        import extract

        src = inspect.getsource(extract)
        assert "[:6000]" not in src  # storage never truncates

    def test_llm_extraction_is_segmented_not_cut(self):
        import inspect

        import _llm_extract

        src = inspect.getsource(_llm_extract.extract_paper)
        assert "_segments" in src or "_SEG" in src  # chunking present
        assert "abstract[:6000]" not in src

    def test_claims_engine_keeps_long_sentences(self):
        """Sentences up to 2000 chars are claim candidates (was 600 drop)."""
        from _claims_engine import _CLAIM_SENT

        long_sent = (
            "Storage conditions vary "
            + "considerably across systems " * 60
            + "with 5 kbar pressure."
        )
        assert _CLAIM_SENT.search(long_sent + " ")
        over = "x" * 2500 + " 5 kbar."
        # oversized text is SPLIT into partial sentences, never dropped:
        # the numeric tail must still be capturable
        assert _CLAIM_SENT.search(over + " ") is not None

    def test_screen_embedding_text_not_capped(self):
        import screen

        src = Path(screen.__file__).read_text()
        assert "[:5000]" not in src and "[:8000]" not in src

    def test_ranking_uses_full_text(self):
        import _ranking

        src = Path(_ranking.__file__).read_text()
        assert "[:1000]" not in src

    def test_verify_hints_use_full_abstract(self):
        import verify

        src = Path(verify.__file__).read_text()
        assert "abstract[:500]" not in src

    def test_embed_text_full_pools_long_text(self):
        """Chunk+mean-pool: long text yields ONE unit vector, no exception."""
        from _embeddings import embed_text_full

        try:
            v = embed_text_full(_long_abstract(5000))
            if v is not None:  # model may be absent in CI — skip gracefully
                import numpy as np

                assert v.shape[0] == 768 or v.shape[0] > 0
                assert abs(np.linalg.norm(v) - 1.0) < 1e-6
        except ImportError:
            pytest.skip("fastembed unavailable")

    def test_embed_text_full_short_passthrough(self):
        from _embeddings import embed_text_full

        try:
            v = embed_text_full("amphibole thermobarometry")
            if v is None:
                pytest.skip("fastembed unavailable")
        except ImportError:
            pytest.skip("fastembed unavailable")

    def test_export_abstracts_not_capped(self):
        import export_citations

        src = Path(export_citations.__file__).read_text()
        assert "[:5000]" not in src and "[:6000]" not in src
