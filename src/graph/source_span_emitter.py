from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, List

from rdflib import URIRef, BNode, Literal
from rdflib.namespace import XSD

from .ontology_builder import OntologyBuilder
from ..models.code_entities import CodeEntity


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".ts", ".tsx"}:
        return "typescript"
    if ext in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    return ext.lstrip('.') or 'unknown'


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / '.git').exists():
            return p
        p = p.parent
    return start.resolve()


def get_git_commit(repo_root: Path) -> str:
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "WORKING"


def repo_relative_path(repo_root: Path, file_path: Path) -> str:
    try:
        return str(file_path.resolve().relative_to(repo_root.resolve()))
    except Exception:
        return str(file_path.resolve())


def make_blob_uri(repo_root: Path, commit: str, file_path: Path) -> str:
    rel = repo_relative_path(repo_root, file_path)
    repo_name = repo_root.name
    return f"blob:git://{repo_name}@{commit}:{rel}"


def compute_byte_offsets(content: bytes, line_start: int, col_start: int, line_end: int, col_end: int) -> Tuple[int, int]:
    # Convert 1-based line numbers to byte offsets
    lines = content.splitlines(keepends=True)
    ls = max(1, line_start) - 1
    le = max(1, line_end) - 1
    byte_start = sum(len(l) for l in lines[:ls]) + max(0, col_start)
    byte_end = sum(len(l) for l in lines[:le]) + max(0, col_end)
    # Clamp
    byte_start = max(0, min(byte_start, len(content)))
    byte_end = max(byte_start, min(byte_end, len(content)))
    return byte_start, byte_end


def emit_blob(builder: OntologyBuilder, file_path: Path, repo_root: Optional[Path] = None) -> URIRef:
    repo_root = repo_root or find_repo_root(file_path)
    commit = get_git_commit(repo_root)
    blob_uri = URIRef(make_blob_uri(repo_root, commit, file_path))

    CODE = builder.CODE
    builder.graph.add((blob_uri, builder.RDF.type, CODE.Blob))
    lang = detect_language(file_path)
    builder.graph.add((blob_uri, CODE.language, Literal(lang, datatype=XSD.string)))

    try:
        data = file_path.read_bytes()
        fh = sha256_hex(data)
        builder.graph.add((blob_uri, CODE.contentHash, Literal(fh, datatype=XSD.string)))
    except Exception:
        pass

    return blob_uri


def emit_source_span_for_entity(builder: OntologyBuilder, entity: CodeEntity, preview_max: int = 300) -> Optional[BNode]:
    try:
        file_path = Path(entity.location.file_path)
        data = file_path.read_bytes()
    except Exception:
        return None

    # Fallbacks for end positions
    line_start = int(entity.location.line_number or 1)
    col_start = int(entity.location.column or 0)
    line_end = int(entity.location.end_line or line_start)
    col_end = int(entity.location.end_column or col_start + 80)

    bstart, bend = compute_byte_offsets(data, line_start, col_start, line_end, col_end)
    slice_bytes = data[bstart:bend]
    content_hash = sha256_hex(slice_bytes)
    # naive AST hash: hash of normalized whitespace
    norm = b" ".join(slice_bytes.split())
    ast_hash = sha256_hex(norm)

    # Ensure blob exists
    blob_uri = emit_blob(builder, file_path)

    CODE = builder.CODE
    span = BNode()
    builder.graph.add((span, builder.RDF.type, CODE.SourceSpan))
    builder.graph.add((URIRef(entity.uri), CODE.hasSourceSpan, span))
    builder.graph.add((span, CODE.inBlob, blob_uri))
    builder.graph.add((span, CODE.byteStart, Literal(bstart, datatype=XSD.integer)))
    builder.graph.add((span, CODE.byteEnd, Literal(bend, datatype=XSD.integer)))
    builder.graph.add((span, CODE.lineStart, Literal(line_start, datatype=XSD.integer)))
    builder.graph.add((span, CODE.lineEnd, Literal(line_end, datatype=XSD.integer)))
    builder.graph.add((span, CODE.contentHash, Literal(content_hash, datatype=XSD.string)))
    builder.graph.add((span, CODE.astHash, Literal(ast_hash, datatype=XSD.string)))

    if slice_bytes:
        try:
            preview = slice_bytes.decode('utf-8', errors='ignore')
            if len(preview) > preview_max:
                preview = preview[:preview_max]
            builder.graph.add((span, CODE.previewText, Literal(preview, datatype=XSD.string)))
        except Exception:
            pass

    return span


def emit_sample_spans(builder: OntologyBuilder, entities: List[CodeEntity], project_root: Path) -> int:
    """Emit sample spans for a few known files if present.
    Returns count of spans emitted.
    """
    targets = [
        project_root / "src" / "app" / "page.js",
        project_root / "src" / "components" / "RichTextEditor.jsx",
    ]
    emitted = 0
    for t in targets:
        if not t.exists():
            continue
        # Pick up to 2 entities declared in this file
        matches = [e for e in entities if Path(e.location.file_path).resolve() == t.resolve()]
        for e in matches[:2]:
            if emit_source_span_for_entity(builder, e):
                emitted += 1
    return emitted

