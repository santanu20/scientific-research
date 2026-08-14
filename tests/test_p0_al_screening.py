"""P0-3 active-learning prioritization tests (reorder-only, sensitivity-first)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _sources import PaperRecord
from screen import load_labels_file, prioritize_active_learning


def _paper(i: int, topic: str) -> PaperRecord:
    return PaperRecord(
        title=f"{topic} study {i}",
        abstract=f"This paper examines {topic} in detail with quantitative methods. "
        f"Study {i} reports measurements of {topic}.",
        doi=f"10.9999/al.{i}",
    )


def _corpus():
    relevant = [_paper(i, "amphibole thermobarometry") for i in range(15)]
    noise = [_paper(100 + i, "clinical trial outcomes") for i in range(15)]
    return relevant, noise


class TestActiveLearning:
    def test_relevant_papers_surface_early(self):
        relevant, noise = _corpus()
        papers = relevant + noise
        labels = {p.doi: "include" for p in relevant[:6]}
        labels.update({p.doi: "exclude" for p in noise[:6]})
        al = prioritize_active_learning(papers, labels)
        assert al.n_seed == 12
        assert al.n_unscreened == 18
        top10_ids = [q["paper_id"] for q in al.queue[:10]]
        # among unscreened relevant (9 left) vs unscreened noise (9 left):
        # top-10 must be majority relevant (recall-first ranking)
        n_rel = sum(1 for pid in top10_ids if int(pid.split(".")[-1]) < 100)
        assert n_rel >= 7, f"expected ≥7 relevant in top-10, got {n_rel}"

    def test_reorder_only_never_drops(self):
        relevant, noise = _corpus()
        papers = relevant + noise
        labels = {p.doi: "include" for p in relevant[:6]}
        labels.update({p.doi: "exclude" for p in noise[:6]})
        al = prioritize_active_learning(papers, labels)
        assert len(al.queue) == len(papers) - 12  # every unscreened paper queued
        assert all(not q["labeled"] for q in al.queue)

    def test_insufficient_seed_fails_loud(self):
        relevant, noise = _corpus()
        labels = {relevant[0].doi: "include", noise[0].doi: "exclude"}
        with pytest.raises(ValueError, match="seed labels"):
            prioritize_active_learning(relevant + noise, labels)

    def test_single_class_seed_fails_loud(self):
        relevant, noise = _corpus()
        labels = {p.doi: "include" for p in relevant[:8]}  # no excludes
        with pytest.raises(ValueError, match="exclude"):
            prioritize_active_learning(relevant + noise, labels)

    def test_maybe_labels_ignored(self):
        relevant, noise = _corpus()
        labels = {p.doi: "include" for p in relevant[:6]}
        labels.update({p.doi: "exclude" for p in noise[:6]})
        labels[relevant[7].doi] = "maybe"  # overridden: was unlabeled anyway
        al = prioritize_active_learning(relevant + noise, labels)
        assert al.n_seed == 12  # maybe not counted

    def test_recall_curve_monotone(self):
        relevant, noise = _corpus()
        labels = {p.doi: "include" for p in relevant[:6]}
        labels.update({p.doi: "exclude" for p in noise[:6]})
        al = prioritize_active_learning(relevant + noise, labels)
        found = [c["predicted_includes"] for c in al.recall_curve]
        assert found == sorted(found)


class TestLabelsFile:
    def test_roundtrip(self, tmp_path):
        f = tmp_path / "labels.jsonl"
        f.write_text(
            '{"doi": "10.1/a", "label": "include"}\n'
            '{"doi": "10.1/b", "label": "exclude"}\n'
            '{"doi": "10.1/c", "label": "maybe"}\n'
            "\n"
        )
        labels = load_labels_file(f)
        assert labels == {"10.1/a": "include", "10.1/b": "exclude", "10.1/c": "maybe"}

    def test_bad_line_raises(self, tmp_path):
        f = tmp_path / "labels.jsonl"
        f.write_text("not json")
        with pytest.raises(ValueError, match="bad labels line"):
            load_labels_file(f)
