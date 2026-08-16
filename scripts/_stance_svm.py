#!/usr/bin/env python3
"""SVM-based stance classifier for scientific-research skill.

Trains a LinearSVC on a bundled 200-sentence corpus (5 scientific fields)
using TF-IDF features (same config as ASReview). Trains on first run (~1s),
caches as pickle, loads instantly on subsequent runs.

Combined with the Jurgens 2018 lexicon for the final classification:
    final_stance = argmax(0.5 * SVM + 0.3 * lexicon + 0.2 * discourse)

Fully automatic — user never provides data. Training data is bundled.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

log = logging.getLogger("scientific_research.stance_svm")

_DATA_PATH = Path(__file__).parent / "data" / "stance_training.jsonl"
_CACHE_PATH = (
    Path(
        os.environ.get(
            "SCIENTIFIC_RESEARCH_CACHE",
            str(Path.home() / ".cache" / "scientific-research"),
        )
    )
    / "svm_stance.pkl"
)

VALID_STANCES = ("supporting", "contrasting", "extending", "methodology", "mentioning")

# Discourse markers (field-agnostic)
_DISCOURSE_SUPPORT = {
    "therefore",
    "thus",
    "hence",
    "furthermore",
    "moreover",
    "consequently",
    "accordingly",
}
_DISCOURSE_CONTRAST = {
    "however",
    "whereas",
    "although",
    "though",
    "nevertheless",
    "nonetheless",
    "conversely",
    "in contrast",
    "on the other hand",
    "despite",
    "in spite of",
    "unlike",
}
_DISCOURSE_METHOD = {
    "we used",
    "we employed",
    "we applied",
    "we measured",
    "we performed",
    "following the",
    "according to the protocol",
}


_model_cache: dict | None = None


def _load_training_data() -> list[dict]:
    """Load bundled stance training data + user extra data (if configured).

    Extra data: set SCIENTIFIC_RESEARCH_STANCE_DATA=/path/to/extra.jsonl with
    one {"text": ..., "label": ..., "field": ...} object per line (same schema
    as data/stance_training.jsonl; "field" optional). Labels MUST be real
    stance annotations — SciCite-style citation-intent labels (background/
    method/result) are NOT stance labels and cannot be auto-mapped without
    fabricating ground truth; convert manually or with a validated mapping.
    Invalid lines are skipped loudly (counted + logged), never silently.
    """
    data = []
    if _DATA_PATH.exists():
        with open(_DATA_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    extra_path = os.environ.get("SCIENTIFIC_RESEARCH_STANCE_DATA", "")
    if extra_path and Path(extra_path).exists():
        added, skipped = 0, 0
        with open(extra_path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                text, label = obj.get("text"), obj.get("label")
                if text and label in VALID_STANCES:
                    data.append(
                        {
                            "text": text,
                            "label": label,
                            "field": obj.get("field", "extra"),
                        }
                    )
                    added += 1
                else:
                    skipped += 1
        log.info(
            "Stance extra data %s: +%d valid, %d skipped (bad label/line; "
            "valid labels: %s)",
            extra_path,
            added,
            skipped,
            ",".join(VALID_STANCES),
        )
    elif extra_path:
        log.warning("SCIENTIFIC_RESEARCH_STANCE_DATA set but missing: %s", extra_path)
    log.debug("Loaded %d stance training sentences", len(data))
    return data


def _train_model() -> dict | None:
    """Train SVM on bundled data. Returns model dict with vectorizer + classifier."""
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
    except ImportError:
        log.warning(
            "scikit-learn not available — SVM stance disabled, using lexicon only"
        )
        return None

    data = _load_training_data()
    if len(data) < 20:
        log.warning("Too few training samples (%d) — SVM disabled", len(data))
        return None

    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    # TF-IDF (same config as ASReview)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)

    # LinearSVC with probability calibration for confidence scores
    base_svc = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, dual="auto")
    clf = CalibratedClassifierCV(base_svc, cv=3)
    clf.fit(X, labels)

    model = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "labels": [str(c) for c in (clf.classes_ or [])],
        "n_training": len(data),
    }

    # Cache
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(
        "SVM stance model trained on %d sentences, cached to %s", len(data), _CACHE_PATH
    )

    return model


def _get_model() -> dict | None:
    """Get trained model (load from cache or train on first run)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    # Try loading cached model
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "rb") as f:
                _model_cache = pickle.load(f)
            log.debug("SVM stance model loaded from cache")
            return _model_cache
        except (pickle.PickleError, OSError) as e:
            log.warning("Failed to load cached SVM model: %s — retraining", e)

    # Train new model
    _model_cache = _train_model()
    return _model_cache


def predict_svm_scores(sentence: str) -> dict[str, float]:
    """Get probability scores for all stance classes.

    Returns dict: {"supporting": 0.7, "contrasting": 0.1, ...}
    """
    model = _get_model()
    if model is None:
        return dict.fromkeys(VALID_STANCES, 0.2)  # uniform fallback

    X = model["vectorizer"].transform([sentence])
    probs = model["classifier"].predict_proba(X)[0]
    classes = model["classifier"].classes_

    scores = dict.fromkeys(VALID_STANCES, 0.1)  # base prior
    for cls, prob in zip(classes, probs):
        scores[str(cls)] = float(prob)
    return scores


def discourse_stance_scores(sentence: str) -> dict[str, float]:
    """Score stance based on discourse markers.

    Field-agnostic: looks for transition words that signal
    support, contrast, or methodology.
    """
    text_lower = " " + sentence.lower() + " "
    scores = dict.fromkeys(VALID_STANCES, 0.0)

    # Support markers
    for marker in _DISCOURSE_SUPPORT:
        if marker in text_lower:
            scores["supporting"] += 0.3
            break

    # Contrast markers
    for marker in _DISCOURSE_CONTRAST:
        if marker in text_lower:
            scores["contrasting"] += 0.3
            break

    # Method markers
    for marker in _DISCOURSE_METHOD:
        if marker in text_lower:
            scores["methodology"] += 0.3
            break

    # Normalize
    total = sum(scores.values())
    if total > 0:
        for k in scores:
            scores[k] = scores[k] / total

    return scores


def combined_stance(
    sentence: str,
    lexicon_stance: str = "",
    lexicon_confidence: float = 0.0,
    lexicon_reason: str = "",
) -> tuple[str, float, str]:
    """Combine SVM + lexicon + discourse for final stance classification.

    Weights: 50% SVM + 30% lexicon + 20% discourse
    Negation override: if lexicon detected negated support → contrast, boost lexicon to 60%.

    Args:
        sentence: the text to classify
        lexicon_stance: stance from Jurgens 2018 lexicon (optional)
        lexicon_confidence: confidence from lexicon (optional)
        lexicon_reason: reason from lexicon (optional — used for negation override)

    Returns: (stance, confidence, reason)
    """
    if not sentence or not sentence.strip():
        return ("mentioning", 0.0, "empty sentence")

    # SVM scores
    svm_scores = predict_svm_scores(sentence)

    # Discourse scores
    disc_scores = discourse_stance_scores(sentence)

    # Lexicon scores (convert single label to distribution)
    lex_scores = dict.fromkeys(VALID_STANCES, 0.1)
    if lexicon_stance and lexicon_stance in lex_scores:
        lex_scores[lexicon_stance] = max(lexicon_confidence, 0.3)

    # Normalize lexicon scores
    lex_total = sum(lex_scores.values())
    if lex_total > 0:
        for k in lex_scores:
            lex_scores[k] /= lex_total

    # Determine weights
    negation_detected = lexicon_reason and "negat" in lexicon_reason.lower()
    svm_max_conf = max(svm_scores.values()) if svm_scores else 0.0
    if negation_detected:
        svm_w, lex_w, disc_w = 0.3, 0.6, 0.1
    elif svm_max_conf < 0.6 and lexicon_stance:
        svm_w, lex_w, disc_w = 0.3, 0.5, 0.2
    else:
        svm_w, lex_w, disc_w = 0.5, 0.3, 0.2

    # Combine: weighted average
    combined = {}
    for s in VALID_STANCES:
        combined[s] = (
            svm_w * svm_scores.get(s, 0.0)
            + lex_w * lex_scores.get(s, 0.0)
            + disc_w * disc_scores.get(s, 0.0)
        )

    # Pick best
    best_stance = max(combined, key=lambda k: combined[k])
    best_score = combined[best_stance]

    # Build reason
    reasons = []
    if svm_scores.get(best_stance, 0) > 0.3:
        reasons.append(f"SVM({svm_scores[best_stance]:.2f})")
    if lex_scores.get(best_stance, 0) > 0.3:
        reasons.append(f"lexicon({lex_scores[best_stance]:.2f})")
    if disc_scores.get(best_stance, 0) > 0.1:
        reasons.append(f"discourse({disc_scores[best_stance]:.2f})")
    if negation_detected:
        reasons.append("negation-override")
    reason = " + ".join(reasons) if reasons else "default"

    return (best_stance, best_score, reason)


def is_available() -> bool:
    """Check if SVM stance classification is available."""
    return _get_model() is not None


def health_check() -> dict:
    """Return health status."""
    model = _get_model()
    data = _load_training_data()
    from collections import Counter

    label_dist = Counter(d["label"] for d in data)
    field_dist = Counter(d.get("field", "?") for d in data)

    return {
        "available": model is not None,
        "training_samples": len(data),
        "label_distribution": dict(label_dist),
        "field_distribution": dict(field_dist),
        "model_cached": _CACHE_PATH.exists(),
        "cache_path": str(_CACHE_PATH),
        "algorithm": "LinearSVC + CalibratedClassifierCV + TF-IDF (ngram 1-2, sublinear)",
    }


if __name__ == "__main__":
    print(json.dumps(health_check(), indent=2))
    print()
    # Test on sample sentences from different fields
    test_sentences = [
        ("Our results confirm the mantle plume hypothesis.", "geology"),
        ("However, our data contradict the standard model.", "physics"),
        ("We extend the catalytic scope to new substrates.", "chemistry"),
        ("We used CRISPR-Cas9 for gene editing.", "biology"),
        ("As cited by Goodfellow et al., deep learning has transformed AI.", "CS"),
    ]
    for sent, field in test_sentences:
        stance, conf, reason = combined_stance(sent)
        print(f"  [{field:<8}] {stance:<14} ({conf:.2f}) {reason}: {sent[:60]}")
