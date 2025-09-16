Code Ontology – Next.js POC
=================================

This project converts codebases into an RDF/OWL knowledge graph with precise structural and semantic relationships. For the Next.js POC, it supports JavaScript/TypeScript projects and integrates AST fallback plus LSP (TypeScript Language Server) when available.

What's New: Source Location Model
- Adds minimal source-location schema with precise spans tied to a file blob at a pinned git commit.
- Classes: `code:Blob`, `code:SourceSpan`, `code:Snippet`
- Object props: `code:hasSourceSpan` (entity → span), `code:inBlob` (span → blob)
- Data props: `code:byteStart`, `code:byteEnd`, `code:lineStart`, `code:lineEnd`, `code:contentHash`, `code:astHash`, `code:language`, `code:previewText`
- Preserves existing `code:locatedAt` and `code:hasBodyHash`.

Hashes
- Uses SHA-256 for `code:contentHash` and `code:astHash`.
- `contentHash` is over the exact byte slice; `astHash` is a normalized (whitespace-collapsed) hash of the same slice.

Blob URIs
- Format: `blob:git://<repo-name>@<commit>:<repo-relative-path>`
- `commit` is from `git rev-parse HEAD`, else `WORKING` if not a git repo.

Saving Ontology
- Saved in Turtle (`.ttl`) under `graph_data/knowledge_graph_<timestamp>.ttl` with `graph_data/latest.ttl` symlink.

Retrieval: SPARQL example
Find blob URI, spans, and hash for a function by name:

```
PREFIX code: <http://codeontology.org/>
SELECT ?blob ?lineStart ?lineEnd ?byteStart ?byteEnd ?contentHash WHERE {
  ?f a code:Function ; code:hasName "normalizeToHtml" ; code:hasSourceSpan ?s .
  ?s code:inBlob ?blob ;
     code:lineStart ?lineStart ; code:lineEnd ?lineEnd ;
     code:byteStart ?byteStart ; code:byteEnd ?byteEnd ;
     code:contentHash ?contentHash .
}
```

Invariants
- URIs are stable across runs for the same commit and path.
- RDF remains lean; no full bodies are in RDF beyond `code:previewText` (truncated to 200–400 chars).
- Full file bytes live in the git working copy or a content-addressed store.

How to Run
1. Optional: install LSP for richer edges
   - `npm i -g typescript-language-server typescript`
2. Process a codebase
   - `source venv/bin/activate`
   - `PYTHONPATH=. python src/pipeline/processor.py /path/to/nextjs rdflib`
3. Output graph
   - `graph_data/latest.ttl`

