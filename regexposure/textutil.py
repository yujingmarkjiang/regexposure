"""Text utilities shared by all steps."""
import hashlib
import html as _html
import json
import re
import threading
from pathlib import Path

from bs4 import BeautifulSoup

BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section",
              "article", "blockquote", "pre", "dt", "dd", "center", "title"]
_ENCODERS = {}


# ------------------------------------------------------------------------------- HTML -> text
def html_to_lines(raw: str) -> list[str]:
    """Convert an HTML (or plain-text) filing to non-empty, whitespace-normalised lines.

    Line breaks are inserted only at block-level elements, so inline spans such as
    <span>Item</span><span>1A.</span> stay on one line. Inline-XBRL headers are removed.
    """
    if not re.search(r"<\s*(html|body|div|p|table|document)\b", raw[:20000], re.I):
        lines = raw.splitlines()
    else:
        soup = BeautifulSoup(raw, "lxml")
        for t in soup(["script", "style", "head"]):
            t.decompose()
        for t in soup.find_all(re.compile(r"^ix:header$", re.I)):
            t.decompose()
        for t in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            t.decompose()
        for t in soup.find_all(BLOCK_TAGS):
            t.append("\n")
        for t in soup.find_all(["td", "th"]):
            t.append(" ")
        lines = soup.get_text("").split("\n")
    out = []
    for line in lines:
        line = re.sub(r"\s+", " ", _html.unescape(line).replace("\xa0", " ")).strip()
        if line:
            out.append(line)
    return out


# ------------------------------------------------------------------------------- tokens
def encoder(name: str):
    import tiktoken
    if name not in _ENCODERS:
        _ENCODERS[name] = tiktoken.get_encoding(name)
    return _ENCODERS[name]


def n_tokens(text: str, encoding: str = "o200k_base") -> int:
    return len(encoder(encoding).encode(text, disallowed_special=()))


def token_chunks(text: str, max_tokens: int, overlap: int = 0, encoding: str = "o200k_base") -> list[str]:
    """Split text into consecutive chunks of at most max_tokens tokens (with optional overlap)."""
    enc = encoder(encoding)
    toks = enc.encode(text, disallowed_special=())
    if len(toks) <= max_tokens:
        return [text]
    step = max(max_tokens - overlap, 1)
    return [enc.decode(toks[i:i + max_tokens]) for i in range(0, len(toks), step) if toks[i:i + max_tokens]]


# ------------------------------------------------------------------------------- hashing / io
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 10) -> str:
    return sha256(text)[:n]


class JsonlStore:
    """Append-only JSONL file used to make every step resumable and traceable."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def records(self):
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def done_ids(self, key="custom_id", ok_only=True):
        return {r[key] for r in self.records() if (r.get("ok", True) or not ok_only)}

    def append(self, rec: dict):
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_text(path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def write_text(path, text: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
