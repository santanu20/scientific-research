"""P1-9 Europe PMC adapter tests (offline — mocked HTTP)."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import httpx
from _sources import _epmc_to_record, epmc_get_by_doi, epmc_search

_CORE_HIT = {
    "title": "Nanometre-scale thermometry in a living cell.",
    "authorString": "Kucsko G, Maurer PC, Yao NY,",
    "abstractText": "We measure temperature in living cells.",
    "doi": "10.1038/nature12373",
    "pmid": 23903748,
    "pubYear": "2013",
    "isOpenAccess": "N",
    "journalInfo": {"journal": {"title": "Nature"}, "yearOfPublication": 2013},
    "pubType": "research-article",
    "language": "eng",
    "keywordList": ["nanodiamond", "thermometry"],
    "id": "3601351",
    "pmcid": "PMC3710522",
}
_NO_ID_HIT = {"title": "No identifiers here", "abstractText": "x"}


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestEpmcToRecord:
    def test_field_mapping(self):
        rec = _epmc_to_record(_CORE_HIT)
        assert rec.doi == "10.1038/nature12373"
        assert rec.pmid == "23903748"
        assert rec.year == 2013
        assert rec.venue == "Nature"
        assert rec.is_open_access is False
        assert rec.source == "epmc"
        assert rec.authors[0] == {"name": "Kucsko G"}
        assert len(rec.authors) == 3
        assert rec.mesh == ["nanodiamond", "thermometry"]
        assert rec.raw_metadata["epmc"]["pmcid"] == "PMC3710522"

    def test_missing_year_is_none(self):
        hit = dict(_CORE_HIT, pubYear="", journalInfo={})
        rec = _epmc_to_record(hit)
        assert rec.year is None


class TestEpmcSearch:
    def test_skips_records_without_identifiers(self):
        payload = {"resultList": {"result": [_CORE_HIT, _NO_ID_HIT]}}
        with patch.object(httpx, "get", return_value=_FakeResp(payload)):
            recs = epmc_search("anything", max_results=10)
        assert len(recs) == 1
        assert recs[0].doi == "10.1038/nature12373"

    def test_http_failure_returns_empty(self):
        with patch.object(httpx, "get", side_effect=ConnectionError("down")):
            recs = epmc_search("anything", max_results=5)
        assert recs == []


class TestEpmcGetByDoi:
    def test_hit(self):
        payload = {"resultList": {"result": [_CORE_HIT]}}
        with patch.object(httpx, "get", return_value=_FakeResp(payload)):
            rec = epmc_get_by_doi("10.1038/nature12373")
        assert rec is not None and rec.title.startswith("Nanometre")

    def test_miss_returns_none(self):
        payload = {"resultList": {"result": []}}
        with patch.object(httpx, "get", return_value=_FakeResp(payload)):
            assert epmc_get_by_doi("10.1/nope") is None
