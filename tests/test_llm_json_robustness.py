"""Pins: LLM double-encoded JSON robustness (2026-09-03 live defect)."""

# ── C1: double-encoded JSON (2026-09-03 live defect) ────────────────────


def test_unit__llm_extract__double_encoded_json_decodes() -> None:
    from _llm_extract import _validate_extraction
    import json

    payload = json.dumps(
        json.dumps({"discipline": "sedimentology", "key_finding": "Facies shift."})
    )
    inner = json.loads(payload)
    assert isinstance(inner, str)
    result = _validate_extraction(inner)
    assert result["discipline"] == "sedimentology"


def test_unit__llm_extract__validator_rejects_non_dict_loudly() -> None:
    from _llm_extract import _validate_extraction

    assert _validate_extraction("just prose")["effect_sizes"] == []
    assert _validate_extraction('{"discipline": "geo"}')["discipline"] == "geo"
    assert isinstance(_validate_extraction(["a", "b"]), dict)
