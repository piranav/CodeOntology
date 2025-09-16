"""
Main pipeline orchestrator for codebase-to-ontology conversion.
Coordinates AST parsing, LSP analysis, and graph population.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json
import hashlib

from ..models.code_entities import CodeEntity, ModuleEntity
from ..parsers.ast_parser import ASTParser, create_ast_parser
from ..parsers.lsp_client import LSPClient, SemanticGraph, create_lsp_client, semantic_graph_to_entities
from ..graph.ontology_builder import OntologyBuilder, create_ontology_builder
from ..graph.graph_store import GraphStore, create_graph_store
from ..graph.query_engine import QueryEngine, create_query_engine
from ..graph.source_span_emitter import emit_sample_spans


@dataclass
class ProcessingStats:
    """Statistics for processing pipeline"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    processing_time: float = 0.0
    entities_created: int = 0
    relationships_created: int = 0
    ast_parsing_time: float = 0.0
    lsp_analysis_time: float = 0.0
    graph_population_time: float = 0.0


@dataclass
class KnowledgeGraph:
    """Complete knowledge graph result"""
    graph_store: GraphStore
    query_engine: QueryEngine
    entities: List[CodeEntity]
    semantic_info: SemanticGraph
    stats: ProcessingStats
    codebase_path: str

    def query(self, query: str, **kwargs):
        """Convenience method for querying"""
        return self.query_engine.sparql_query(query, **kwargs)

    def get_context(self, entity_uri: str, depth: int = 2):
        """Get contextual information"""
        return self.query_engine.get_context(entity_uri, depth)

    def save(self, format_type: str = "turtle") -> bool:
        """Save the knowledge graph"""
        return self.graph_store.save(format_type)


class CodebaseProcessor:
    """
    Main orchestrator for codebase analysis and knowledge graph construction
    Implements the core pipeline: Code → AST + LSP → Knowledge Graph
    """

    def __init__(self,
                 max_workers: int = 4,
                 backend_type: str = 'rdflib',
                 ontology_path: Optional[str] = None,
                 storage_path: Optional[str] = None):

        # Configuration
        self.max_workers = max_workers
        self.backend_type = backend_type
        self.ontology_path = ontology_path
        self.storage_path = storage_path

        # Components (initialized lazily)
        self._ast_parser: Optional[ASTParser] = None
        self._lsp_client: Optional[LSPClient] = None
        self._ontology_builder: Optional[OntologyBuilder] = None
        self._graph_store: Optional[GraphStore] = None
        self._query_engine: Optional[QueryEngine] = None

        # Processing state
        self._file_hashes: Dict[str, str] = {}
        self._processed_entities: Dict[str, CodeEntity] = {}

    @property
    def ast_parser(self) -> ASTParser:
        """Lazy initialization of AST parser"""
        if self._ast_parser is None:
            self._ast_parser = create_ast_parser()
        return self._ast_parser

    @property
    def lsp_client(self) -> LSPClient:
        """Lazy initialization of LSP client"""
        if self._lsp_client is None:
            self._lsp_client = create_lsp_client()
        return self._lsp_client

    @property
    def ontology_builder(self) -> OntologyBuilder:
        """Lazy initialization of ontology builder"""
        if self._ontology_builder is None:
            self._ontology_builder = create_ontology_builder(self.ontology_path)
        return self._ontology_builder

    @property
    def graph_store(self) -> GraphStore:
        """Lazy initialization of graph store"""
        if self._graph_store is None:
            self._graph_store = create_graph_store(self.backend_type, self.storage_path)
        return self._graph_store

    @property
    def query_engine(self) -> QueryEngine:
        """Lazy initialization of query engine"""
        if self._query_engine is None:
            self._query_engine = create_query_engine(self.graph_store)
        return self._query_engine

    def process_codebase(self, codebase_path: str,
                        incremental: bool = False) -> KnowledgeGraph:
        """
        Main entry point for codebase processing
        Returns complete knowledge graph
        """
        start_time = time.time()
        stats = ProcessingStats()

        print(f"🚀 Starting codebase analysis: {codebase_path}")
        print(f"📊 Backend: {self.backend_type}, Workers: {self.max_workers}")

        try:
            # Step 1: Discover and analyze files
            source_files = self.discover_files(codebase_path)
            stats.total_files = len(source_files)

            print(f"📁 Found {stats.total_files} source files")

            if not source_files:
                raise ValueError(f"No source files found in {codebase_path}")

            # Step 2: Load existing graph if incremental
            if incremental:
                print("🔄 Loading existing graph for incremental update")
                self.graph_store.load()

            # Step 3: Parse AST for all files (parallel)
            print("🌳 Parsing AST...")
            ast_start = time.time()
            all_entities = self._parse_ast_parallel(source_files, stats)
            stats.ast_parsing_time = time.time() - ast_start

            print(f"✅ AST parsing complete: {stats.entities_created} entities")

            # Step 4: Perform semantic analysis with LSP
            print("🔍 Performing semantic analysis...")
            lsp_start = time.time()
            semantic_graph = asyncio.run(self._analyze_semantics(codebase_path, all_entities))
            stats.lsp_analysis_time = time.time() - lsp_start

            print(f"✅ Semantic analysis complete: {len(semantic_graph.references)} references")

            # Step 5: Enhance entities with semantic information
            enhanced_entities = semantic_graph_to_entities(semantic_graph, all_entities)

            # Step 6: Populate knowledge graph
            print("📚 Building knowledge graph...")
            graph_start = time.time()
            success = self.graph_store.add_entities_from_list(enhanced_entities)
            stats.graph_population_time = time.time() - graph_start

            if not success:
                raise RuntimeError("Failed to populate knowledge graph")

            # Emit sample source spans for selected files if present
            try:
                emitted = emit_sample_spans(self.ontology_builder, enhanced_entities, Path(codebase_path))
                if emitted:
                    print(f"🧩 Emitted {emitted} sample source spans")
                # Merge builder graph with store's RDFLib graph if using rdflib backend
                if self.backend_type == 'rdflib':
                    # Merge span triples into the existing RDF graph
                    target = self.graph_store.backend.graph
                    for t in self.ontology_builder.graph:
                        target.add(t)
            except Exception as e:
                print(f"⚠️  Failed to emit sample spans: {e}")

            # Step 7: Infer additional patterns and relationships
            print("🧠 Inferring patterns...")
            self._infer_patterns(enhanced_entities, semantic_graph)

            # Step 8: Validate graph consistency
            print("✅ Validating graph...")
            validation_issues = self._validate_graph()
            if validation_issues:
                print(f"⚠️  Validation issues found: {len(validation_issues)}")
                for issue in validation_issues[:5]:  # Show first 5
                    print(f"   - {issue}")

            # Final statistics
            stats.processing_time = time.time() - start_time
            stats.processed_files = len([f for f in source_files if self._get_file_hash(f) in self._file_hashes])
            graph_stats = self.graph_store.get_statistics()
            stats.relationships_created = graph_stats.get('edges', 0) if self.backend_type == 'networkx' else graph_stats.get('total_triples', 0)

            print(f"🎉 Processing complete in {stats.processing_time:.2f}s")
            self._print_final_stats(stats, graph_stats)

            return KnowledgeGraph(
                graph_store=self.graph_store,
                query_engine=self.query_engine,
                entities=enhanced_entities,
                semantic_info=semantic_graph,
                stats=stats,
                codebase_path=codebase_path
            )

        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise

    def discover_files(self, codebase_path: str) -> List[Path]:
        """
        Discover all source files in codebase
        Respects gitignore and common ignore patterns
        """
        codebase_path = Path(codebase_path).resolve()
        source_files = []

        # Supported extensions (focused on JavaScript/TypeScript)
        supported_extensions = {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}

        # Common directories to ignore
        ignore_dirs = {
            'node_modules', 'dist', 'build', '.git', '.next', 'coverage',
            '__pycache__', '.pytest_cache', 'venv', 'env', '.vscode', '.idea'
        }

        # Common files to ignore
        ignore_files = {
            '.gitignore', '.npmrc', '.yarnrc', 'package-lock.json', 'yarn.lock',
            '.env', '.env.local', '.env.production'
        }

        try:
            # Walk directory tree
            for root, dirs, files in os.walk(codebase_path):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]

                for file in files:
                    if file in ignore_files or file.startswith('.'):
                        continue

                    file_path = Path(root) / file

                    # Check extension
                    if file_path.suffix in supported_extensions:
                        # Additional filters
                        if self._should_include_file(file_path):
                            source_files.append(file_path)

        except Exception as e:
            print(f"Error discovering files: {e}")

        return sorted(source_files)  # Sort for consistent processing order

    def _should_include_file(self, file_path: Path) -> bool:
        """Additional file filtering logic"""

        # Skip if file is too large (>1MB for now)
        try:
            if file_path.stat().st_size > 1024 * 1024:  # 1MB
                print(f"⚠️  Skipping large file: {file_path}")
                return False
        except:
            return False

        # Skip generated files
        generated_indicators = ['.generated.', '.min.', '.bundle.', 'vendor']
        filename = file_path.name.lower()

        for indicator in generated_indicators:
            if indicator in filename:
                return False

        # Skip test files for now (could be included with flag)
        test_indicators = ['.test.', '.spec.', '__tests__']
        for indicator in test_indicators:
            if indicator in filename or indicator in str(file_path):
                return False

        return True

    def _parse_ast_parallel(self, source_files: List[Path], stats: ProcessingStats) -> List[CodeEntity]:
        """Parse AST for all files in parallel"""
        all_entities = []

        def parse_single_file(file_path: Path) -> Tuple[Path, List[CodeEntity], bool]:
            """Parse single file and return results"""
            try:
                # Check if file changed (for incremental processing)
                current_hash = self._get_file_hash(file_path)
                if str(file_path) in self._file_hashes and self._file_hashes[str(file_path)] == current_hash:
                    # File unchanged, skip
                    return file_path, [], True

                # Parse file
                root_node, entities = self.ast_parser.parse_file(str(file_path))

                # Update file hash
                self._file_hashes[str(file_path)] = current_hash

                return file_path, entities, True

            except Exception as e:
                print(f"❌ Failed to parse {file_path}: {e}")
                return file_path, [], False

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(parse_single_file, file_path): file_path
                for file_path in source_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    _, entities, success = future.result()

                    if success:
                        all_entities.extend(entities)
                        stats.processed_files += 1
                        stats.entities_created += len(entities)

                        # Store entities by URI for later reference
                        for entity in entities:
                            self._processed_entities[entity.uri] = entity
                    else:
                        stats.failed_files += 1

                except Exception as e:
                    print(f"❌ Processing error for {file_path}: {e}")
                    stats.failed_files += 1

        return all_entities

    async def _analyze_semantics(self, codebase_path: str, entities: List[CodeEntity]) -> SemanticGraph:
        """Perform semantic analysis using LSP"""
        try:
            # Analyze workspace with LSP
            semantic_graph = await self.lsp_client.analyze_workspace(codebase_path)

            return semantic_graph

        except Exception as e:
            print(f"⚠️  LSP analysis failed, using fallback: {e}")

            # Return minimal semantic graph
            return SemanticGraph()

    def _infer_patterns(self, entities: List[CodeEntity], semantic_graph: SemanticGraph):
        """Infer additional relationships and architectural patterns"""

        print("🔄 Inferring architectural patterns...")

        try:
            # Infer design patterns
            self._infer_design_patterns(entities)

            # Infer dependency relationships
            self._infer_dependencies(entities)

            # Infer API boundaries
            self._infer_api_boundaries(entities)

            print("✅ Pattern inference complete")

        except Exception as e:
            print(f"⚠️  Pattern inference failed: {e}")

    def _infer_design_patterns(self, entities: List[CodeEntity]):
        """Detect common design patterns"""

        # Simple pattern detection (can be enhanced)
        patterns_found = []

        # Singleton pattern detection
        for entity in entities:
            if hasattr(entity, 'name') and 'singleton' in entity.name.lower():
                patterns_found.append(('singleton', entity.uri))

        # Factory pattern detection
        for entity in entities:
            if hasattr(entity, 'name') and ('factory' in entity.name.lower() or 'create' in entity.name.lower()):
                patterns_found.append(('factory', entity.uri))

        print(f"   Found {len(patterns_found)} design pattern instances")

    def _infer_dependencies(self, entities: List[CodeEntity]):
        """Infer dependency relationships between modules"""

        module_deps = {}

        for entity in entities:
            if isinstance(entity, ModuleEntity):
                # Analyze imports to build dependency graph
                deps = set()
                for import_uri in entity.imports:
                    # Extract module from import URI
                    if import_uri in self._processed_entities:
                        import_entity = self._processed_entities[import_uri]
                        if hasattr(import_entity, 'module_path'):
                            deps.add(import_entity.module_path)

                module_deps[entity.location.file_path] = deps

        print(f"   Analyzed dependencies for {len(module_deps)} modules")

    def _infer_api_boundaries(self, entities: List[CodeEntity]):
        """Identify API boundaries and public interfaces"""

        public_apis = []

        for entity in entities:
            # Functions/classes that are exported are potential API boundaries
            if hasattr(entity, 'is_exported') and entity.is_exported:
                public_apis.append(entity.uri)

        print(f"   Identified {len(public_apis)} public API endpoints")

    def _validate_graph(self) -> List[str]:
        """Validate knowledge graph for consistency"""

        issues = []

        try:
            if self.backend_type == 'rdflib':
                # Use ontology builder validation
                issues = self.ontology_builder.validate_graph()

            # Additional custom validations
            stats = self.graph_store.get_statistics()

            # Check for reasonable entity ratios
            if self.backend_type == 'networkx':
                node_count = stats.get('nodes', 0)
                edge_count = stats.get('edges', 0)

                if node_count > 0 and edge_count / node_count < 0.5:
                    issues.append(f"Low relationship density: {edge_count}/{node_count}")

        except Exception as e:
            issues.append(f"Validation error: {e}")

        return issues

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content for change detection"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()[:16]
        except:
            return ""

    def _print_final_stats(self, stats: ProcessingStats, graph_stats: Dict[str, Any]):
        """Print comprehensive processing statistics"""

        print("\n📊 Processing Statistics:")
        print(f"   Files processed: {stats.processed_files}/{stats.total_files}")
        print(f"   Files failed: {stats.failed_files}")
        print(f"   Entities created: {stats.entities_created}")
        print(f"   Relationships: {stats.relationships_created}")
        print(f"   Total time: {stats.processing_time:.2f}s")
        print(f"   AST parsing: {stats.ast_parsing_time:.2f}s")
        print(f"   LSP analysis: {stats.lsp_analysis_time:.2f}s")
        print(f"   Graph building: {stats.graph_population_time:.2f}s")

        if 'entity_counts' in graph_stats:
            print("\n📈 Entity Distribution:")
            for entity_type, count in graph_stats['entity_counts'].items():
                print(f"   {entity_type}: {count}")

        # Performance metrics
        if stats.total_files > 0:
            print(f"\n⚡ Performance:")
            print(f"   Files/second: {stats.processed_files / stats.processing_time:.1f}")
            print(f"   Entities/second: {stats.entities_created / stats.processing_time:.1f}")

    def save_graph(self, format_type: str = "turtle", filepath: Optional[str] = None) -> bool:
        """Save the knowledge graph"""
        if filepath:
            # Custom filepath logic would go here
            pass

        return self.graph_store.save(format_type)

    def load_graph(self, filepath: Optional[str] = None) -> bool:
        """Load existing knowledge graph"""
        return self.graph_store.load(filepath)

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of last processing run"""
        return {
            'stats': self.graph_store.get_statistics(),
            'file_hashes': len(self._file_hashes),
            'processed_entities': len(self._processed_entities),
            'backend_type': self.backend_type
        }

    def cleanup(self):
        """Cleanup resources"""
        if self._lsp_client:
            self._lsp_client.shutdown()


# Factory function
def create_codebase_processor(max_workers: int = 4,
                             backend_type: str = 'rdflib',
                             ontology_path: Optional[str] = None,
                             storage_path: Optional[str] = None) -> CodebaseProcessor:
    """Create codebase processor with specified configuration"""
    return CodebaseProcessor(max_workers, backend_type, ontology_path, storage_path)


# CLI interface for testing
def main():
    """Main CLI interface for testing the processor"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python processor.py <codebase_path> [backend_type]")
        sys.exit(1)

    codebase_path = sys.argv[1]
    backend_type = sys.argv[2] if len(sys.argv) > 2 else 'rdflib'

    processor = create_codebase_processor(backend_type=backend_type)

    try:
        knowledge_graph = processor.process_codebase(codebase_path)

        # Save the graph
        success = knowledge_graph.save()
        print(f"Graph saved: {success}")

        # Example queries
        print("\n🔍 Sample Queries:")

        # Query 1: Find all functions
        result = knowledge_graph.query_engine.sparql_query("""
        SELECT ?function ?name WHERE {
            ?function a code:Function .
            ?function code:hasName ?name .
        } LIMIT 5
        """)

        print(f"Functions found: {result.total_results}")
        for r in result.results[:5]:
            print(f"  - {r.get('name', 'unnamed')}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main()
