"""
LSP client integration for semantic analysis.
Captures cross-references, type information, and call graphs.

Design goals:
- Pure-stdio JSON-RPC client so we don't depend on GUI or editors
- Pluggable language adapters (start/stop server, normalize responses)
- Graceful fallback when servers aren't available
"""

import asyncio
import json
import subprocess
import tempfile
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Minimal JSON-RPC message framing over stdio per LSP spec
class JsonRpcStdioClient:
    def __init__(self, cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None):
        self.cmd = cmd
        self.cwd = str(cwd) if cwd else None
        self.env = env or os.environ.copy()
        self.proc: Optional[subprocess.Popen] = None
        self._id_counter = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._loop = asyncio.get_event_loop()

    async def start(self):
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            text=False,
            bufsize=0,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self._loop.run_in_executor(None, self.proc.wait), timeout=3)
                except asyncio.TimeoutError:
                    self.proc.kill()
        finally:
            self.proc = None
            if self._reader_task:
                self._reader_task.cancel()

    async def notify(self, method: str, params: Any):
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._write_message(message)

    async def request(self, method: str, params: Any, timeout: float = 10.0) -> Any:
        self._id_counter += 1
        req_id = self._id_counter
        fut: asyncio.Future = self._loop.create_future()
        self._pending[req_id] = fut
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        await self._write_message(message)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)

    async def _write_message(self, message: Dict[str, Any]):
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("LSP process not started")
        data = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        self.proc.stdin.write(header + data)
        self.proc.stdin.flush()

    async def _read_loop(self):
        if not self.proc or not self.proc.stdout:
            return
        stdout = self.proc.stdout
        while True:
            try:
                header = b""
                while not header.endswith(b"\r\n\r\n"):
                    chunk = stdout.read(1)
                    if not chunk:
                        await asyncio.sleep(0.01)
                        continue
                    header += chunk
                try:
                    header_text = header.decode("ascii")
                except Exception:
                    continue
                content_length = 0
                for line in header_text.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":", 1)[1].strip())
                        break
                if content_length <= 0:
                    continue
                body = stdout.read(content_length)
                if not body:
                    continue
                message = json.loads(body.decode("utf-8"))
                # Server→client request
                if isinstance(message, dict) and message.get("id") is not None and message.get("method"):
                    method = message.get("method")
                    mid = message.get("id")
                    result = None
                    if method == "workspace/configuration":
                        result = [{} for _ in message.get("params", {}).get("items", [])]
                    elif method == "workspace/workspaceFolders":
                        result = []
                    elif method == "client/registerCapability":
                        result = None
                    elif method == "window/showMessageRequest":
                        result = {"title": "OK"}
                    elif method == "workspace/workDoneProgress/create":
                        result = None
                    # Send minimal response
                    resp = {"jsonrpc": "2.0", "id": mid, "result": result}
                    await self._write_message(resp)
                # Response to our request
                elif isinstance(message, dict) and message.get("id") is not None:
                    req_id = message["id"]
                    fut = self._pending.get(req_id)
                    if fut and not fut.done():
                        if "result" in message:
                            fut.set_result(message["result"])
                        else:
                            fut.set_exception(RuntimeError(str(message.get("error"))))
                else:
                    # Notifications (no response required)
                    m = message.get("method")
                    # Common notifications: logMessage, $/progress, publishDiagnostics
                    # We intentionally ignore these but could capture diagnostics/logs if needed
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.01)

from ..models.code_entities import (
    SourceLocation, CallExpressionEntity, FunctionEntity,
    VariableEntity, TypeInfo
)


@dataclass
class SemanticReference:
    """Represents a semantic reference between code entities"""
    source_uri: str  # may be a URI or a symbolic key to be resolved later
    target_uri: str  # may be a URI or a symbolic key to be resolved later
    reference_type: str  # 'calls', 'references', 'defines', 'implements'
    location: SourceLocation
    context: Optional[str] = None


@dataclass
class TypeReference:
    """Type information from LSP"""
    symbol_name: str
    type_name: str
    location: SourceLocation
    is_definition: bool = False
    documentation: Optional[str] = None


@dataclass
class CallHierarchyInfo:
    """Call hierarchy information"""
    function_uri: str
    callers: List[str]  # URIs of functions that call this
    callees: List[str]  # URIs of functions this calls
    call_sites: List[SourceLocation]


class SemanticGraph:
    """Container for semantic analysis results"""

    def __init__(self):
        self.references: List[SemanticReference] = []
        self.type_references: List[TypeReference] = []
        self.call_hierarchy: Dict[str, CallHierarchyInfo] = {}
        self.symbol_definitions: Dict[str, SourceLocation] = {}
        self.cross_file_references: List[SemanticReference] = []

    def add_reference(self, ref: SemanticReference):
        """Add a semantic reference"""
        self.references.append(ref)

        # Track cross-file references
        source_file = ref.location.file_path
        target_location = self.symbol_definitions.get(ref.target_uri)
        if target_location and target_location.file_path != source_file:
            self.cross_file_references.append(ref)

    def get_references_to(self, target_uri: str) -> List[SemanticReference]:
        """Get all references to a specific entity"""
        return [ref for ref in self.references if ref.target_uri == target_uri]

    def get_references_from(self, source_uri: str) -> List[SemanticReference]:
        """Get all references made by a specific entity"""
        return [ref for ref in self.references if ref.source_uri == source_uri]


class LanguageAdapter:
    """Base class for language-specific server integration."""

    name: str = "generic"
    extensions: Tuple[str, ...] = ()

    def __init__(self, workspace: Path, command: List[str]):
        self.workspace = workspace
        self.command = command
        self.client = JsonRpcStdioClient(command, cwd=workspace)
        self.initialized = False

    async def start(self):
        await self.client.start()
        # Initialize
        init_params = {
            "processId": os.getpid(),
            "rootUri": self.workspace.as_uri(),
            "rootPath": str(self.workspace),
            "clientInfo": {"name": "code-ontology", "version": "1.0"},
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didSave": True, "willSave": False, "dynamicRegistration": False},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                    "references": {},
                    "definition": {},
                    "callHierarchy": {}
                },
                "workspace": {"workspaceFolders": True}
            },
            "trace": "off",
            "workspaceFolders": [{"uri": self.workspace.as_uri(), "name": self.workspace.name}],
        }
        try:
            await self.client.request("initialize", init_params, timeout=15)
            await self.client.notify("initialized", {})
            self.initialized = True
        except Exception as e:
            await self.client.stop()
            raise e

    async def stop(self):
        try:
            await self.client.notify("exit", None)
        finally:
            await self.client.stop()

    async def did_open(self, file_path: Path, language_id: str, text: str):
        await self.client.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": file_path.resolve().as_uri(),
                "languageId": language_id,
                "version": 1,
                "text": text,
            }
        })

    async def document_symbol(self, file_path: Path) -> Any:
        return await self.client.request("textDocument/documentSymbol", {
            "textDocument": {"uri": file_path.resolve().as_uri()}
        })

    async def references(self, file_path: Path, position: Tuple[int, int]) -> Any:
        return await self.client.request("textDocument/references", {
            "textDocument": {"uri": file_path.resolve().as_uri()},
            "position": {"line": position[0], "character": position[1]},
            "context": {"includeDeclaration": True}
        })

    async def definition(self, file_path: Path, position: Tuple[int, int]) -> Any:
        return await self.client.request("textDocument/definition", {
            "textDocument": {"uri": file_path.resolve().as_uri()},
            "position": {"line": position[0], "character": position[1]},
        })

    async def call_hierarchy_prepare(self, file_path: Path, position: Tuple[int, int]) -> Any:
        return await self.client.request("textDocument/prepareCallHierarchy", {
            "textDocument": {"uri": file_path.resolve().as_uri()},
            "position": {"line": position[0], "character": position[1]},
        })

    async def call_hierarchy_incoming(self, item: Dict[str, Any]) -> Any:
        return await self.client.request("callHierarchy/incomingCalls", {"item": item})

    async def call_hierarchy_outgoing(self, item: Dict[str, Any]) -> Any:
        return await self.client.request("callHierarchy/outgoingCalls", {"item": item})


class TypeScriptAdapter(LanguageAdapter):
    name = "typescript"
    extensions = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")

    def __init__(self, workspace: Path, command: Optional[List[str]] = None):
        cmd = command or self._resolve_command(workspace)
        super().__init__(workspace, cmd)

    def language_id_for(self, file_path: Path) -> str:
        ext = file_path.suffix
        if ext in (".ts", ".tsx"):
            return "typescript"
        if ext in (".js", ".jsx", ".mjs", ".cjs"):
            return "javascript"
        return "javascript"

    def _resolve_command(self, workspace: Path) -> List[str]:
        # Try direct in PATH
        try:
            r = subprocess.run(["typescript-language-server", "--version"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                return ["typescript-language-server", "--stdio"]
        except Exception:
            pass
        # Try local node_modules
        local_bin = workspace / "node_modules" / ".bin" / "typescript-language-server"
        if local_bin.exists():
            return [str(local_bin), "--stdio"]
        # Fallback to npx
        return ["npx", "typescript-language-server", "--stdio"]


class LSPClient:
    """
    Language Server Protocol client for semantic analysis
    Supports TypeScript/JavaScript language servers
    """

    def __init__(self):
        self.workspace_roots: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.adapters: Dict[str, LanguageAdapter] = {}

    async def analyze_workspace(self, workspace_path: str) -> SemanticGraph:
        """
        Analyze entire workspace using LSP
        Returns semantic graph with all relationships
        """
        workspace_path = Path(workspace_path).resolve()
        self.workspace_roots.add(str(workspace_path))

        # Initialize semantic graph
        semantic_graph = SemanticGraph()

        # Discover source files
        source_files = self._discover_source_files(workspace_path)

        if not source_files:
            print(f"No source files found in {workspace_path}")
            return semantic_graph

        # Use TypeScript adapter for TS/JS
        ts_files = [f for f in source_files if f.suffix in ('.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs')]
        if ts_files:
            try:
                await self._analyze_with_typescript(workspace_path, ts_files, semantic_graph)
            except Exception as e:
                print(f"Error analyzing TypeScript/JavaScript files: {e}")

        return semantic_graph

    def _discover_source_files(self, workspace_path: Path) -> List[Path]:
        """Discover all source files in workspace"""
        source_files = []

        # Common source extensions
        extensions = {'.js', '.jsx', '.ts', '.tsx'}

        # Walk directory tree
        for root, dirs, files in os.walk(workspace_path):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and
                      d not in {'node_modules', 'dist', 'build', '__pycache__'}]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in extensions:
                    source_files.append(file_path)

        return source_files

    async def _analyze_with_typescript(self, workspace_path: Path, files: List[Path], semantic_graph: SemanticGraph):
        # We will attempt to start via adapter which resolves command, no pre-check required

        adapter = TypeScriptAdapter(workspace_path)
        await adapter.start()
        try:
            # Open all docs
            for f in files:
                try:
                    text = f.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    text = ""
                await adapter.did_open(f, adapter.language_id_for(f), text)

            # Build symbol index
            file_symbols: Dict[Path, List[Dict[str, Any]]] = {}
            for f in files:
                try:
                    symbols = await adapter.document_symbol(f)
                    flat = self._flatten_symbols(symbols)
                    file_symbols[f] = flat
                    self._ingest_document_symbols(semantic_graph, f, flat)
                except Exception as e:
                    print(f"documentSymbol failed for {f}: {e}")

            # Attempt call hierarchy for functions/methods
            for f, symbols in file_symbols.items():
                for s in symbols:
                    if s.get('kind') in (12, 11, 6, 5, 'Function', 'Method', 'Constructor'):  # SymbolKind mapping
                        # LSP is 0-based positions
                        pos = (max(0, int(s.get('line', 0)) - 1), int(s.get('character', 0)))
                        try:
                            prepared = await adapter.call_hierarchy_prepare(f, pos)
                        except Exception:
                            prepared = None
                        if isinstance(prepared, list):
                            for item in prepared:
                                # Outgoing calls
                                try:
                                    outgoing = await adapter.call_hierarchy_outgoing(item)
                                except Exception:
                                    outgoing = []
                                for oc in (outgoing or []):
                                    target = oc.get('to', {})
                                    tname = target.get('name') or ''
                                    turi = target.get('uri') or f.as_uri()
                                    trng = target.get('range', {}).get('start', {})
                                    skey_src = self._symbol_key(str(f), s.get('name',''), s.get('line',0), s.get('character',0))
                                    skey_tgt = self._symbol_key(self._uri_to_path(turi), tname, int(trng.get('line',0))+1, int(trng.get('character',0)))
                                    semantic_graph.add_reference(SemanticReference(
                                        source_uri=skey_src,
                                        target_uri=skey_tgt,
                                        reference_type='calls',
                                        location=SourceLocation(file_path=str(f), line_number=s.get('line',0), column=s.get('character',0))
                                    ))
                        # References for symbol usages
                        try:
                            refs = await adapter.references(f, pos)
                        except Exception:
                            refs = []
                        for r in (refs or []):
                            loc = r.get('uri') or f.as_uri()
                            rng = r.get('range', {}).get('start', {})
                            skey_src = self._symbol_key(self._uri_to_path(loc), s.get('name',''), int(rng.get('line',0))+1, int(rng.get('character',0)))
                            skey_tgt = self._symbol_key(str(f), s.get('name',''), s.get('line',0), s.get('character',0))
                            semantic_graph.add_reference(SemanticReference(
                                source_uri=skey_src,
                                target_uri=skey_tgt,
                                reference_type='references',
                                location=SourceLocation(file_path=self._uri_to_path(loc), line_number=int(rng.get('line',0))+1, column=int(rng.get('character',0)))
                            ))
        finally:
            await adapter.stop()

    async def _check_command_available(self, cmd: List[str]) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    # ----------------------
    # Symbol ingestion logic
    # ----------------------
    def _ingest_document_symbols(self, semantic_graph: SemanticGraph, file_path: Path, symbols: Any):
        """Normalize DocumentSymbols or SymbolInformation[] into the semantic graph's definition index."""
        try:
            for s in (symbols or []):
                if 'name' not in s:
                    continue
                line = int(s.get('line', 0))
                col = int(s.get('character', 0))
                location = SourceLocation(file_path=str(file_path), line_number=line, column=col)
                key = self._symbol_key(str(file_path), s['name'], line, col)
                semantic_graph.symbol_definitions[key] = location
        except Exception:
            pass

    def _flatten_symbols(self, symbols: Any) -> List[Dict[str, Any]]:
        flat: List[Dict[str, Any]] = []
        if not isinstance(symbols, list):
            return flat
        def add_sym(s, parent_kind=None):
            name = s.get('name')
            if not name:
                return
            if 'range' in s:
                start = s['range'].get('start', {})
                line = int(start.get('line', 0)) + 1
                col = int(start.get('character', 0))
                kind = s.get('kind', parent_kind)
                flat.append({'name': name, 'line': line, 'character': col, 'kind': kind})
                for c in s.get('children', []) or []:
                    add_sym(c, parent_kind=kind)
            elif 'location' in s:
                rng = s['location'].get('range', {}).get('start', {})
                line = int(rng.get('line', 0)) + 1
                col = int(rng.get('character', 0))
                flat.append({'name': name, 'line': line, 'character': col, 'kind': s.get('kind')})
        for s in symbols:
            add_sym(s)
        return flat

    def _symbol_key(self, file_path: str, name: str, line: int, col: int) -> str:
        return f"KEY|{file_path}|{name}|{line}:{col}"

    def _uri_to_path(self, uri: str) -> str:
        if uri.startswith('file://'):
            from urllib.parse import urlparse, unquote
            p = urlparse(uri)
            return unquote(p.path)
        return uri

    # Legacy simplified single-file analysis retained as fallback API

    async def _initialize_lsp_connection(self, client: object, workspace_path: Path):
        """Initialize LSP connection (simplified)"""
        # In a full implementation, this would:
        # 1. Send initialize request
        # 2. Handle initialize response
        # 3. Send initialized notification
        # For now, we'll use a simplified approach
        pass

    async def _analyze_single_file(
        self,
        client: object,
        file_path: Path,
        semantic_graph: SemanticGraph
    ):
        """Analyze a single file for semantic information"""

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create document identifier
        doc_uri = file_path.as_uri()
        doc_id = TextDocumentIdentifier(uri=doc_uri)

        # Since we can't easily integrate full LSP here, we'll use a simplified approach
        # that extracts the most important semantic information
        await self._extract_semantic_info_simplified(file_path, content, semantic_graph)

    async def _extract_semantic_info_simplified(
        self,
        file_path: Path,
        content: str,
        semantic_graph: SemanticGraph
    ):
        """
        Simplified semantic analysis without full LSP integration
        Extracts basic cross-references and type information
        """

        lines = content.split('\n')
        file_uri = str(file_path)

        # Extract import/export relationships
        await self._extract_import_export_relationships(lines, file_path, semantic_graph)

        # Extract function calls
        await self._extract_function_calls(lines, file_path, semantic_graph)

        # Extract variable references
        await self._extract_variable_references(lines, file_path, semantic_graph)

    async def _extract_import_export_relationships(
        self,
        lines: List[str],
        file_path: Path,
        semantic_graph: SemanticGraph
    ):
        """Extract import/export semantic relationships"""

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Import statements
            if line.startswith('import'):
                # Parse import statement
                if 'from' in line:
                    # import { symbol } from 'module'
                    parts = line.split('from')
                    if len(parts) == 2:
                        module_part = parts[1].strip().strip('"\';')

                        # Try to resolve module path
                        resolved_path = self._resolve_module_path(module_part, file_path)

                        if resolved_path:
                            ref = SemanticReference(
                                source_uri=str(file_path),
                                target_uri=str(resolved_path),
                                reference_type='imports',
                                location=SourceLocation(
                                    file_path=str(file_path),
                                    line_number=line_num,
                                    column=0
                                ),
                                context=line
                            )
                            semantic_graph.add_reference(ref)

            # Export statements
            elif line.startswith('export'):
                # Track what this file exports
                symbol_name = self._extract_export_symbol(line)
                if symbol_name:
                    location = SourceLocation(
                        file_path=str(file_path),
                        line_number=line_num,
                        column=0
                    )
                    semantic_graph.symbol_definitions[f"{file_path}#{symbol_name}"] = location

    async def _extract_function_calls(
        self,
        lines: List[str],
        file_path: Path,
        semantic_graph: SemanticGraph
    ):
        """Extract function call relationships"""

        for line_num, line in enumerate(lines, 1):
            # Simple regex-like extraction for function calls
            # In practice, this would use AST + LSP for accuracy

            # Look for function call patterns: functionName(
            import re
            call_pattern = r'(\w+)\s*\('

            for match in re.finditer(call_pattern, line):
                function_name = match.group(1)

                # Skip keywords
                if function_name in {'if', 'for', 'while', 'switch', 'catch', 'function', 'class'}:
                    continue

                # Create call reference
                ref = SemanticReference(
                    source_uri=str(file_path),
                    target_uri=f"function:{function_name}",  # Would be resolved to actual URI
                    reference_type='calls',
                    location=SourceLocation(
                        file_path=str(file_path),
                        line_number=line_num,
                        column=match.start()
                    ),
                    context=line.strip()
                )
                semantic_graph.add_reference(ref)

    async def _extract_variable_references(
        self,
        lines: List[str],
        file_path: Path,
        semantic_graph: SemanticGraph
    ):
        """Extract variable reference relationships"""

        # Track variable declarations and usage
        declared_vars = set()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Variable declarations
            if any(line.startswith(keyword) for keyword in ['const ', 'let ', 'var ']):
                # Extract variable name
                import re
                var_match = re.search(r'(const|let|var)\s+(\w+)', line)
                if var_match:
                    var_name = var_match.group(2)
                    declared_vars.add(var_name)

                    # Record definition
                    location = SourceLocation(
                        file_path=str(file_path),
                        line_number=line_num,
                        column=var_match.start(2)
                    )
                    semantic_graph.symbol_definitions[f"{file_path}#{var_name}"] = location

            # Variable usage (simplified)
            for var_name in declared_vars:
                if var_name in line and not line.startswith(('const ', 'let ', 'var ')):
                    ref = SemanticReference(
                        source_uri=str(file_path),
                        target_uri=f"{file_path}#{var_name}",
                        reference_type='references',
                        location=SourceLocation(
                            file_path=str(file_path),
                            line_number=line_num,
                            column=line.find(var_name)
                        ),
                        context=line
                    )
                    semantic_graph.add_reference(ref)

    def _resolve_module_path(self, module_path: str, current_file: Path) -> Optional[Path]:
        """Resolve relative module path to absolute path"""

        if module_path.startswith('.'):
            # Relative import
            resolved = (current_file.parent / module_path).resolve()

            # Try different extensions
            for ext in ['.js', '.jsx', '.ts', '.tsx', '/index.js', '/index.ts']:
                candidate = Path(str(resolved) + ext)
                if candidate.exists():
                    return candidate

            # Try as directory with index
            if resolved.is_dir():
                for index_file in ['index.js', 'index.ts', 'index.jsx', 'index.tsx']:
                    candidate = resolved / index_file
                    if candidate.exists():
                        return candidate

        # Node modules or absolute imports would need more complex resolution
        return None

    def _extract_export_symbol(self, export_line: str) -> Optional[str]:
        """Extract symbol name from export statement"""
        import re

        # export function name()
        func_match = re.search(r'export\s+function\s+(\w+)', export_line)
        if func_match:
            return func_match.group(1)

        # export class Name
        class_match = re.search(r'export\s+class\s+(\w+)', export_line)
        if class_match:
            return class_match.group(1)

        # export const name
        const_match = re.search(r'export\s+const\s+(\w+)', export_line)
        if const_match:
            return const_match.group(1)

        # export default
        if 'export default' in export_line:
            return 'default'

        return None

    async def get_call_hierarchy(self, function_uri: str) -> CallHierarchyInfo:
        """Get call hierarchy for a function (simplified implementation)"""

        # This would use LSP call hierarchy in a full implementation
        # For now, return empty hierarchy
        return CallHierarchyInfo(
            function_uri=function_uri,
            callers=[],
            callees=[],
            call_sites=[]
        )

    async def get_references(self, symbol_uri: str, workspace_path: str) -> List[SemanticReference]:
        """Get all references to a symbol (simplified implementation)"""

        # This would use LSP textDocument/references in a full implementation
        return []

    async def get_definition(self, symbol_name: str, location: SourceLocation) -> Optional[SourceLocation]:
        """Get definition location of symbol (simplified implementation)"""

        # This would use LSP textDocument/definition in a full implementation
        return None

    async def get_type_information(self, location: SourceLocation) -> Optional[TypeReference]:
        """Get type information at location (simplified implementation)"""

        # This would use LSP textDocument/hover or typeDefinition in a full implementation
        return None

    def shutdown(self):
        """Shutdown all language server processes"""
        # Legacy shutdown compatibility
        try:
            if hasattr(self, 'client_processes') and isinstance(self.client_processes, dict):
                for process in self.client_processes.values():
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                self.client_processes.clear()
        except Exception:
            pass
        try:
            self.executor.shutdown(wait=True)
        except Exception:
            pass

    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown()


# Factory function
def create_lsp_client() -> LSPClient:
    """Create LSP client instance"""
    return LSPClient()


# Utility functions for integration with ontology builder
def semantic_graph_to_entities(semantic_graph: SemanticGraph, base_entities: List) -> List:
    """
    Convert semantic graph information back to entity relationships
    This bridges the gap between LSP analysis and ontology population
    """

    # Build indices to resolve symbolic keys to entity URIs
    entity_map = {entity.uri: entity for entity in base_entities}
    index_by_file_and_name: Dict[Tuple[str, str], List] = {}
    for e in base_entities:
        key = (e.location.file_path, e.name)
        index_by_file_and_name.setdefault(key, []).append(e)

    def resolve(key_or_uri: str):
        if key_or_uri in entity_map:
            return entity_map[key_or_uri]
        # Try symbolic key: KEY|file|name|line:col
        if key_or_uri.startswith('KEY|'):
            try:
                _, fpath, name, pos = key_or_uri.split('|', 3)
                line = int(pos.split(':')[0])
            except Exception:
                fpath, name, line = None, None, None
            if fpath and name:
                candidates = index_by_file_and_name.get((fpath, name), [])
                if candidates:
                    # choose best by closest line
                    return sorted(candidates, key=lambda c: abs((c.location.line_number or 0) - (line or 0)))[0]
        return None

    for ref in semantic_graph.references:
        source_entity = resolve(ref.source_uri)
        target_entity = resolve(ref.target_uri)

        if source_entity and target_entity:
            if ref.reference_type == 'calls':
                if hasattr(source_entity, 'calls'):
                    if target_entity.uri not in source_entity.calls:
                        source_entity.calls.append(target_entity.uri)
                if hasattr(target_entity, 'called_by'):
                    if source_entity.uri not in target_entity.called_by:
                        target_entity.called_by.append(source_entity.uri)

            elif ref.reference_type == 'references':
                if hasattr(source_entity, 'accesses_variables'):
                    if target_entity.uri not in source_entity.accesses_variables:
                        source_entity.accesses_variables.append(target_entity.uri)
                if hasattr(target_entity, 'used_by'):
                    if source_entity.uri not in target_entity.used_by:
                        target_entity.used_by.append(source_entity.uri)

    return base_entities
