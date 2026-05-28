"""Build a small petroleum-engineering corpus for a RAG workshop.

The script downloads a curated list of open educational/government pages and
PDFs, extracts clean text, chunks it, and writes JSONL files for a RAG notebook.

Outputs:
  data/petroleum_corpus/raw/        downloaded HTML/PDF files
  data/petroleum_corpus/documents.jsonl
  data/petroleum_corpus/chunks.jsonl
  data/petroleum_corpus/corpus_report.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import requests  # type: ignore[import-not-found]
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError as exc:
    BeautifulSoup = None

try:
    from pypdf import PdfReader
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pypdf\n"
        "Install dependencies with: python -m pip install -r requirements.txt\n"
        "If you use the Codex bundled Python runtime, pypdf is already available."
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):  # type: ignore[no-redef]
        return iterable


DEFAULT_SOURCES = Path("./data/source/petroleum_corpus_sources.json")
DEFAULT_OUT_DIR = Path("./data/petroleum_corpus")
USER_AGENT = (
    "Mozilla/5.0 (compatible; petroleum-rag-corpus-builder/1.0; "
    "+https://example.invalid/educational-workshop)"
)


@dataclass(frozen=True)
class Source:
    topic: str
    title: str
    url: str
    source: str
    license_note: str = ""


def load_sources(path: Path) -> list[Source]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Source(**row) for row in rows]


def slugify(value: str, max_len: int = 80) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:max_len].strip("-") or "document"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def fetch_url(session: object | None, url: str, timeout: int = 45) -> tuple[bytes, str]:
    if requests is not None and session is not None:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        return response.content, content_type

    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "").lower()
        return response.read(), content_type


class BasicHTMLTextExtractor(HTMLParser):
    """Small fallback extractor for educational pages when bs4 is unavailable."""

    block_tags = {"h1", "h2", "h3", "h4", "p", "li", "td", "th"}
    skip_tags = {"script", "style", "noscript", "svg", "form", "iframe", "nav", "header", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self._current_tag: str | None = None
        self._buffer: list[str] = []
        self.blocks: list[str] = []
        self.title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.skip_tags:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if self._skip_depth == 0 and tag in self.block_tags:
            self._current_tag = tag
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.skip_tags and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if self._skip_depth == 0 and tag == self._current_tag:
            text = normalize_text(" ".join(self._buffer))
            if len(text) >= 25 or tag in {"h1", "h2", "h3", "h4"}:
                self.blocks.append(text)
            self._current_tag = None
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data.strip() + " "
        if self._current_tag:
            self._buffer.append(data)


def extract_html_text(html_bytes: bytes, fallback_title: str) -> tuple[str, str]:
    if BeautifulSoup is None:
        parser = BasicHTMLTextExtractor()
        parser.feed(html_bytes.decode("utf-8", errors="ignore"))
        title = normalize_text(parser.title) or fallback_title
        return title, normalize_text("\n\n".join(parser.blocks))

    soup = BeautifulSoup(html_bytes, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "form", "iframe"]):
        tag.decompose()
    for selector in ["nav", "header", "footer", ".breadcrumb", ".menu", ".tabs"]:
        for tag in soup.select(selector):
            tag.decompose()

    title = fallback_title
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(" ", strip=True)

    main = soup.find("main") or soup.find("article") or soup.body or soup
    blocks: list[str] = []
    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
        text = tag.get_text(" ", strip=True)
        if len(text) < 25 and tag.name not in {"h1", "h2", "h3", "h4"}:
            continue
        if text.lower() in {"print", "share", "menu", "search"}:
            continue
        blocks.append(text)

    text = normalize_text("\n\n".join(blocks))
    return title, text


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {idx}]\n{page_text}")
    return normalize_text("\n\n".join(pages))


def detect_kind(url: str, content_type: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf") or "application/pdf" in content_type:
        return "pdf"
    return "html"


def chunk_text(
    text: str,
    chunk_words: int = 350,
    overlap_words: int = 60,
) -> list[dict[str, object]]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_words_list = words[start:end]
        chunk = " ".join(chunk_words_list)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "word_start": start,
                "word_end": end,
                "word_count": len(chunk_words_list),
                "text": chunk,
            }
        )
        if end == len(words):
            break
        start = max(0, end - overlap_words)
        chunk_index += 1
    return chunks


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_report(out_dir: Path, docs: list[dict[str, object]], chunks: list[dict[str, object]]) -> None:
    total_words = sum(int(doc["word_count"]) for doc in docs)
    avg_doc_words = total_words / len(docs) if docs else 0
    avg_chunk_words = (
        sum(int(chunk["word_count"]) for chunk in chunks) / len(chunks) if chunks else 0
    )

    by_topic: dict[str, int] = {}
    for doc in docs:
        by_topic[str(doc["topic"])] = by_topic.get(str(doc["topic"]), 0) + 1

    lines = [
        "# Petroleum RAG Corpus Report",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Documents: {len(docs)}",
        f"- Chunks: {len(chunks)}",
        f"- Total words: {total_words}",
        f"- Mean document words: {avg_doc_words:.1f}",
        f"- Mean chunk words: {avg_chunk_words:.1f}",
        "",
        "## Documents",
        "",
        "| Topic | Title | Words | URL |",
        "|---|---:|---:|---|",
    ]

    for doc in docs:
        title = str(doc["title"]).replace("|", "\\|")
        lines.append(
            f"| {doc['topic']} | {title} | {doc['word_count']} | {doc['url']} |"
        )

    lines.extend(["", "## Topic Coverage", ""])
    for topic, count in sorted(by_topic.items()):
        lines.append(f"- {topic}: {count} document(s)")

    lines.extend(
        [
            "",
            "## Suggested Notebook Diagnostics",
            "",
            "- Corpus statistics table: documents, chunks, mean/median chunk length.",
            "- Retrieval check: for each test question, show top-3 retrieved chunks with source URL.",
            "- Metric: recall@k using `expected_answer_keywords` from `rag_test_questions.json`.",
            "- Heatmap: cosine similarity between each test question embedding and each document or chunk embedding.",
        ]
    )

    (out_dir / "corpus_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build petroleum RAG corpus.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--chunk-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=60)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = load_sources(args.sources)
    if args.limit:
        sources = sources[: args.limit]

    if args.dry_run:
        for source in sources:
            print(f"{source.topic}: {source.title} <{source.url}>")
        return 0

    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = None
    if requests is not None:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

    documents: list[dict[str, object]] = []
    all_chunks: list[dict[str, object]] = []
    accessed_at = dt.datetime.now(dt.timezone.utc).isoformat()

    for source in tqdm(sources, desc="Downloading sources"):
        try:
            payload, content_type = fetch_url(session, source.url)
            kind = detect_kind(source.url, content_type)
            digest = sha256_bytes(payload)
            stem = f"{slugify(source.topic)}-{digest[:10]}"
            raw_path = raw_dir / f"{stem}.{kind}"
            raw_path.write_bytes(payload)

            if kind == "pdf":
                title = source.title
                text = extract_pdf_text(payload)
            else:
                title, text = extract_html_text(payload, source.title)

            wc = word_count(text)
            if wc < 80:
                print(f"Skipping short extraction ({wc} words): {source.url}", file=sys.stderr)
                continue

            doc_id = stem
            doc = {
                "doc_id": doc_id,
                "topic": source.topic,
                "title": title,
                "url": source.url,
                "source": source.source,
                "license_note": source.license_note,
                "kind": kind,
                "raw_path": str(raw_path.as_posix()),
                "sha256": digest,
                "accessed_at": accessed_at,
                "word_count": wc,
                "text": text,
            }
            documents.append(doc)

            for chunk in chunk_text(text, args.chunk_words, args.overlap_words):
                chunk_id = f"{doc_id}_chunk_{chunk['chunk_index']:04d}"
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "topic": source.topic,
                        "title": title,
                        "url": source.url,
                        "chunk_index": chunk["chunk_index"],
                        "word_start": chunk["word_start"],
                        "word_end": chunk["word_end"],
                        "word_count": chunk["word_count"],
                        "text": chunk["text"],
                    }
                )

            time.sleep(args.delay_seconds)
        except Exception as exc:
            print(f"Failed: {source.url}\n  {type(exc).__name__}: {exc}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "/clean/documents.jsonl", documents)
    write_jsonl(args.out_dir / "/clean/chunks.jsonl", all_chunks)
    build_report(args.out_dir, documents, all_chunks)

    print(f"Wrote {len(documents)} documents and {len(all_chunks)} chunks to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
