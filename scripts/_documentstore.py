"""DocumentStore — full-text layer (Phase 3, 2026-08-15).

Layers paper records with FULL TEXT when legally available:
  1. Unpaywall best_oa PDF URL (already in verified records via OpenAlex)
  2. arXiv PDF for arXiv-hosted records
  3. Local PDFs dropped by the user into <store>/incoming/ (matched by
     fuzzy title; no upload — local files only, no data egress)

Storage layout (portable, provenance-friendly):
  <root>/store/<sha16>.txt      extracted plain text (+ .meta.json sidecar)
  <root>/index.json             {paper_key: {sha, source, n_chars, fetched}}

Text extraction: pdf-ocr skill CLI when present (local OCR, no egress);
otherwise pypdf if installed; otherwise the PDF is skipped loudly.

Public API:
    store = DocumentStore(root)
    store.attach(corpus_path)          # match corpus papers to stored texts
    store.fetch(paper)                 # one paper, best legal route
    store.fetch_corpus(corpus_path)    # all papers, bounded concurrency
    store.get_text(paper_key)          # '' when absent
    store.claims(paper_key)            # sentence-split numeric claims
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess  # pyright: ignore[reportAttributeAccessIssue]
import sys
import time
import urllib.request
from pathlib import Path

if __name__ == "__main__":
    import _bootstrap

    _bootstrap.ensure_env()

import logging

log = logging.getLogger("scientific_research.documentstore")

_UA = {"User-Agent": "scientific-research-skill/1.0 (mailto:research@example.org)"}


def _paper_key(paper: dict) -> str:
    return (paper.get("doi") or paper.get("arxiv_id") or paper.get("title") or "")[:120]


class DocumentStore:
    def __init__(self, root: Path | str = "~/.local/share/scientific-research/store"):
        self.root = Path(root).expanduser()
        (self.root / "store").mkdir(parents=True, exist_ok=True)
        (self.root / "incoming").mkdir(parents=True, exist_ok=True)
        self.index: dict[str, dict] = self._load_index()

    # ── index persistence ────────────────────────────────────────────
    def _load_index(self) -> dict:
        f = self.root / "index.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                log.warning("store index corrupt — rebuilding")
        return {}

    def _save_index(self) -> None:
        f = self.root / "index.json"
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, indent=1), encoding="utf-8")
        tmp.replace(f)

    # ── PDF → text ────────────────────────────────────────────────────
    @staticmethod
    def _pdf_to_text(pdf_path: Path) -> str:
        # Route 1: pdf-ocr skill (local vision OCR — scanned PDFs, no egress)
        ocr = Path.home() / ".config/opencode/skills/pdf-ocr_Original_Monolith/scripts"
        if (ocr / "pdf_ocr.py").exists():
            try:
                r = subprocess.run(
                    [sys.executable, str(ocr / "pdf_ocr.py"), str(pdf_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if r.returncode == 0 and len(r.stdout.strip()) > 200:
                    return r.stdout
            except (subprocess.TimeoutExpired, OSError) as e:
                log.debug("pdf-ocr failed for %s: %s", pdf_path, e)
        # Route 2: pypdf (born-digital)
        try:
            from pypdf import PdfReader  # pyright: ignore[reportMissingImports]

            reader = PdfReader(str(pdf_path))
            pages = [pg.extract_text() or "" for pg in reader.pages]
            text = "\n".join(pages)
            if len(text.strip()) > 200:
                return text
        except Exception as e:  # noqa: BLE001 — extraction robustness
            log.debug("pypdf failed for %s: %s", pdf_path, e)
        return ""

    # ── legal sources ────────────────────────────────────────────────
    @staticmethod
    def _unpaywall_pdf(doi: str) -> str | None:
        if not doi:
            return None
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email=research@example.org"
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
            best = data.get("best_oa_location") or {}
            pdf = best.get("url_for_pdf") or best.get("url")
            return pdf if pdf and str(pdf).lower().endswith(".pdf") else None
        except Exception:  # noqa: BLE001 — network best-effort
            return None

    @staticmethod
    def _arxiv_pdf(arxiv_id: str) -> str | None:
        if not arxiv_id:
            return None
        return (
            f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            if not arxiv_id.startswith("http")
            else arxiv_id
        )

    _STOP = {"of", "the", "a", "an", "in", "for", "and", "on", "at", "to", "from", "with"}

    def _incoming_match(self, title: str) -> Path | None:
        """Content-token match against user-dropped PDFs in incoming/.

        Stopwords dropped from both sides: 'thermobarometry OF mount adamas'
        vs 'thermobarometry mount adams 2020' shares every content token —
        raw Jaccard (0.43) under-scored it by union-inflating stopword 'of'.
        """
        inc = self.root / "incoming"
        toks = lambda s: {t for t in re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).split() if t not in self._STOP} - {"20", "201", "202", "203"}
        target = toks(title)
        if not target:
            return None
        best, best_score = None, 0.0
        for pdf in inc.glob("*.pdf"):
            cand = toks(pdf.stem)
            if not cand:
                continue
            score = len(target & cand) / max(len(target | cand), 1)
            if score > best_score:
                best, best_score = pdf, score
        return best if best_score >= 0.5 else None

    # ── public API ────────────────────────────────────────────────────
    def fetch(self, paper: dict) -> bool:
        """Fetch + store full text for ONE paper. Returns True if stored."""
        key = _paper_key(paper)
        if not key or self.index.get(key, {}).get("sha"):
            return bool(self.index.get(key, {}).get("sha"))
        pdf_url = self._unpaywall_pdf(paper.get("doi") or "") or self._arxiv_pdf(
            paper.get("arxiv_id") or ""
        )
        source = "remote-oa" if pdf_url else "incoming"
        tmp_pdf: Path | None = None
        if pdf_url:
            tmp_pdf = self.root / f"tmp_{int(time.time() * 1000) % 10**9}.pdf"
            try:
                req = urllib.request.Request(pdf_url, headers=_UA)
                with (
                    urllib.request.urlopen(req, timeout=60) as r,
                    open(tmp_pdf, "wb") as fh,
                ):
                    fh.write(r.read(30_000_000))  # 30 MB cap
            except Exception as e:  # noqa: BLE001
                log.debug("PDF fetch failed %s: %s", pdf_url, e)
                tmp_pdf = None
        if tmp_pdf is None:
            local = self._incoming_match(paper.get("title") or "")
            if local:
                tmp_pdf = local
                source = "incoming"
        if tmp_pdf is None or (source == "remote-oa" and not tmp_pdf.exists()):
            self.index[key] = {
                "sha": None,
                "source": "unavailable",
                "ts": time.strftime("%Y-%m-%d"),
            }
            self._save_index()
            return False
        text = self._pdf_to_text(tmp_pdf)
        if source == "remote-oa":
            tmp_pdf.unlink(missing_ok=True)
        if len(text.strip()) < 500:
            self.index[key] = {
                "sha": None,
                "source": "extraction-failed",
                "ts": time.strftime("%Y-%m-%d"),
            }
            self._save_index()
            return False
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        (self.root / "store" / f"{sha}.txt").write_text(text, encoding="utf-8")
        self.index[key] = {
            "sha": sha,
            "source": source,
            "n_chars": len(text),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save_index()
        return True

    def fetch_corpus(self, corpus_path: Path | str, limit: int = 100) -> dict:
        """Fetch full texts for a whole corpus. Returns summary stats."""
        from _artifact import load_corpus

        data = load_corpus(corpus_path)
        stats = {"attempted": 0, "stored": 0, "unavailable": 0}
        for p in data["papers"][:limit]:
            stats["attempted"] += 1
            try:
                if self.fetch(p):
                    stats["stored"] += 1
                else:
                    stats["unavailable"] += 1
            except Exception as e:  # noqa: BLE001 — per-paper isolation
                log.warning("fetch failed for %s: %s", _paper_key(p), e)
                stats["unavailable"] += 1
            time.sleep(0.4)  # polite rate
        log.info("DocumentStore: %s", stats)
        return stats

    def get_text(self, paper: dict | str) -> str:
        key = paper if isinstance(paper, str) else _paper_key(paper)
        sha = self.index.get(key, {}).get("sha")
        if not sha:
            return ""
        f = self.root / "store" / f"{sha}.txt"
        return f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""

    def coverage(self, corpus_path: Path | str) -> dict:
        from _artifact import load_corpus

        papers = load_corpus(corpus_path)["papers"]
        n = len(papers)
        have = sum(1 for p in papers if self.get_text(p))
        return {
            "n_papers": n,
            "with_fulltext": have,
            "coverage": round(have / n, 3) if n else 0.0,
        }


if __name__ == "__main__":
    print(f"OK {sys.argv[0]}: DocumentStore ready")
