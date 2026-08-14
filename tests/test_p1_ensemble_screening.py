"""P1-10 LLM ensemble screening tests (offline — mocked LLM layer)."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class _Paper:
    def __init__(
        self, title="Deep learning mineral mapping", abstract="We map minerals."
    ):
        self.title = title
        self.abstract = abstract
        self.raw_metadata = {}


class TestEnsembleLogic:
    def test_majority_vote_include(self):
        import _llm_extract as lx

        def fake_judge(paper, query, model=None, research_type=None):
            # two models say yes, one says no
            verdicts = {
                "m1": (True, "relevant"),
                "m2": (True, "relevant"),
                "m3": (False, "off-topic"),
            }
            return verdicts.get(model)

        with (
            patch.object(lx, "_pick_model", return_value="m1"),
            patch.object(
                lx, "_detect_models", return_value=[{"name": "m2"}, {"name": "m3"}]
            ),
            patch.object(lx, "llm_screen_paper", side_effect=fake_judge),
        ):
            decision, reason, votes = lx.llm_screen_paper_ensemble(
                _Paper(), "mineral mapping"
            )
        assert decision is True
        assert "2/3 judges" in reason
        assert len(votes) == 3

    def test_majority_vote_exclude(self):
        import _llm_extract as lx

        def fake_judge(paper, query, model=None, research_type=None):
            verdicts = {"m1": (False, "off"), "m2": (False, "off"), "m3": (True, "on")}
            return verdicts.get(model)

        with (
            patch.object(lx, "_pick_model", return_value="m1"),
            patch.object(
                lx, "_detect_models", return_value=[{"name": "m2"}, {"name": "m3"}]
            ),
            patch.object(lx, "llm_screen_paper", side_effect=fake_judge),
        ):
            decision, reason, votes = lx.llm_screen_paper_ensemble(
                _Paper(), "clinical trial"
            )
        assert decision is False
        assert len(votes) == 3

    def test_single_model_degrades_honestly(self):
        import _llm_extract as lx

        with (
            patch.object(lx, "_pick_model", return_value="solo"),
            patch.object(lx, "_detect_models", return_value=[]),
            patch.object(lx, "llm_screen_paper", return_value=(True, "relevant")),
        ):
            decision, reason, votes = lx.llm_screen_paper_ensemble(_Paper(), "q")
        assert decision is True
        assert "single judge" in reason and "ensemble unavailable" in reason
        assert len(votes) == 1

    def test_no_llm_returns_none(self):
        import _llm_extract as lx

        with (
            patch.object(lx, "_pick_model", return_value=None),
        ):
            assert lx.llm_screen_paper_ensemble(_Paper(), "q") is None

    def test_vote_failure_excluded_not_fatal(self):
        import _llm_extract as lx

        def fake_judge(paper, query, model=None, research_type=None):
            if model == "m2":
                return None  # one judge down
            return (True, "ok")

        with (
            patch.object(lx, "_pick_model", return_value="m1"),
            patch.object(
                lx, "_detect_models", return_value=[{"name": "m2"}, {"name": "m3"}]
            ),
            patch.object(lx, "llm_screen_paper", side_effect=fake_judge),
        ):
            decision, _reason, votes = lx.llm_screen_paper_ensemble(_Paper(), "q")
        assert decision is True
        assert len(votes) == 2


class TestScreenPaperWiring:
    def test_ensemble_stage_label(self):
        import _llm_extract as lx
        import screen_llm

        with (
            patch.object(lx, "_pick_model", return_value="m1"),
            patch.object(lx, "_detect_models", return_value=[{"name": "m2"}]),
            patch.object(lx, "llm_screen_paper", return_value=(True, "relevant")),
        ):
            decision, stage, reason = screen_llm.screen_paper(
                _Paper(), "mineral mapping", use_llm=True, ensemble=True
            )
        assert decision is True
        assert stage == "llm_ensemble"

    def test_single_judge_path_unchanged(self):
        import _llm_extract as lx
        import screen_llm

        with patch.object(lx, "llm_screen_paper", return_value=(False, "off-topic")):
            decision, stage, _ = screen_llm.screen_paper(
                _Paper(), "clinical trial", use_llm=True
            )
        assert decision is False
        assert stage == "llm_judge"

    def test_judge_none_falls_permissive(self):
        import _llm_extract as lx
        import screen_llm

        with patch.object(lx, "llm_screen_paper", return_value=None):
            decision, stage, _ = screen_llm.screen_paper(_Paper(), "q", use_llm=True)
        assert decision is True
        assert stage == "llm_unavailable"
