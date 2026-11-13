from __future__ import annotations

from pathlib import Path
import re
from .models import Chunk

SUPPORTED = {".txt", ".md", ".pdf"}


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("Install PyMuPDF to read PDF files: pip install pymupdf") from exc
        with fitz.open(path) as pdf:
            return "\n\n".join(page.get_text("text") for page in pdf)
    return path.read_text(encoding="utf-8", errors="replace")


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                segment = paragraph[start:end].strip()
                if segment:
                    chunks.append(segment)
                if end == len(paragraph):
                    break
                start = max(start + 1, end - overlap_chars)
            continue
        proposed = f"{current}\n\n{paragraph}" if current else paragraph
        if len(proposed) <= max_chars:
            current = proposed
        else:
            if current:
                chunks.append(current)
            overlap = current[-overlap_chars:] if current and overlap_chars else ""
            current = f"{overlap}\n{paragraph}".strip() if overlap else paragraph
            if len(current) > max_chars:
                current = current[-max_chars:]
    if current:
        chunks.append(current)
    return chunks


def load_documents(data_dir: str | Path, *, max_chars: int = 1400, overlap_chars: int = 180) -> list[Chunk]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    chunks: list[Chunk] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED):
        doc_id = path.stem
        pieces = _split_text(_extract_text(path), max_chars, overlap_chars)
        for index, text in enumerate(pieces, start=1):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::{index:04d}",
                doc_id=doc_id,
                text=text,
                source=path.name,
                metadata={"path": str(path), "chunk_index": index},
            ))
    if not chunks:
        raise ValueError(f"No supported documents found in {root}; supported: {sorted(SUPPORTED)}")
    return chunks
