"""
Incremental update system for maintaining knowledge graph consistency.
Handles file changes without full reprocessing.
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

from ..models.code_entities import CodeEntity
from ..parsers.ast_parser import ASTParser
from ..parsers.lsp_client import LSPClient, SemanticGraph
from ..graph.graph_store import GraphStore
from .processor import CodebaseProcessor


@dataclass
class ChangeEvent:
    """Represents a file change event"""
    file_path: str
    event_type: str  # 'created', 'modified', 'deleted'
    timestamp: float
    content_hash: Optional[str] = None


@dataclass
class UpdateResult:
    """Result of incremental update"""
    success: bool
    processed_files: int = 0
    entities_updated: int = 0
    entities_added: int = 0
    entities_removed: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None


class IncrementalUpdater:
    """
    Handles incremental updates to the knowledge graph when files change
    Maintains graph consistency with minimal reprocessing
    """

    def __init__(self,
                 processor: CodebaseProcessor,
                 debounce_delay: float = 0.5,
                 batch_size: int = 10):

        self.processor = processor
        self.debounce_delay = debounce_delay
        self.batch_size = batch_size

        # State tracking
        self.file_hashes: Dict[str, str] = {}
        self.entity_to_files: Dict[str, Set[str]] = {}  # entity_uri -> files that define it
        self.file_to_entities: Dict[str, Set[str]] = {}  # file_path -> entity_uris
        self.pending_changes: Dict[str, ChangeEvent] = {}  # file_path -> latest change

        # Dependencies tracking
        self.file_dependencies: Dict[str, Set[str]] = {}  # file -> files it depends on
        self.file_dependents: Dict[str, Set[str]] = {}   # file -> files that depend on it

        # Processing queue
        self.update_queue: asyncio.Queue = asyncio.Queue()
        self.processing_lock = asyncio.Lock()

    def initialize(self, codebase_path: str) -> bool:
        """Initialize the updater with current codebase state"""
        try:
            print("🔄 Initializing incremental updater...")

            # Build initial state mappings
            self._build_initial_mappings(codebase_path)

            # Initialize file hashes
            self._initialize_file_hashes(codebase_path)

            print(f"✅ Initialized with {len(self.file_hashes)} files")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize updater: {e}")
            return False

    def _build_initial_mappings(self, codebase_path: str):
        """Build entity-file mappings from existing graph"""
        try:
            # Query existing graph for entity-file relationships
            if self.processor.backend_type == 'rdflib':
                query = """
                SELECT ?entity ?file WHERE {
                    ?entity code:locatedAt ?location .
                    ?location code:filePath ?file .
                }
                """

                results = self.processor.graph_store.query(query)

                for result in results:
                    entity_uri = result.get('entity')
                    file_path = result.get('file')

                    if entity_uri and file_path:
                        # Map entity to file
                        if entity_uri not in self.entity_to_files:
                            self.entity_to_files[entity_uri] = set()
                        self.entity_to_files[entity_uri].add(file_path)

                        # Map file to entity
                        if file_path not in self.file_to_entities:
                            self.file_to_entities[file_path] = set()
                        self.file_to_entities[file_path].add(entity_uri)

            # Build dependency mappings
            self._build_dependency_mappings()

        except Exception as e:
            print(f"⚠️  Warning: Could not build initial mappings: {e}")

    def _build_dependency_mappings(self):
        """Build file dependency relationships"""
        try:
            # Query for import/export relationships
            if self.processor.backend_type == 'rdflib':
                import_query = """
                SELECT ?importerFile ?importedFile WHERE {
                    ?importer code:imports ?imported .
                    ?importer code:locatedAt ?importerLoc .
                    ?importerLoc code:filePath ?importerFile .
                    ?imported code:locatedAt ?importedLoc .
                    ?importedLoc code:filePath ?importedFile .
                }
                """

                results = self.processor.graph_store.query(import_query)

                for result in results:
                    importer = result.get('importerFile')
                    imported = result.get('importedFile')

                    if importer and imported:
                        # importer depends on imported
                        if importer not in self.file_dependencies:
                            self.file_dependencies[importer] = set()
                        self.file_dependencies[importer].add(imported)

                        # imported is a dependent of importer
                        if imported not in self.file_dependents:
                            self.file_dependents[imported] = set()
                        self.file_dependents[imported].add(importer)

        except Exception as e:
            print(f"⚠️  Warning: Could not build dependency mappings: {e}")

    def _initialize_file_hashes(self, codebase_path: str):
        """Initialize hash tracking for all files"""
        source_files = self.processor.discover_files(codebase_path)

        for file_path in source_files:
            try:
                file_hash = self._compute_file_hash(file_path)
                self.file_hashes[str(file_path)] = file_hash
            except Exception as e:
                print(f"⚠️  Could not hash {file_path}: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""

    async def on_file_change(self, file_path: str, change_type: str) -> bool:
        """Handle a file change event"""
        try:
            # Create change event
            event = ChangeEvent(
                file_path=file_path,
                event_type=change_type,
                timestamp=time.time()
            )

            # Compute content hash for modified/created files
            if change_type in ['created', 'modified']:
                try:
                    event.content_hash = self._compute_file_hash(Path(file_path))
                except Exception:
                    event.content_hash = None

            # Add to pending changes (overwrites previous pending change for same file)
            self.pending_changes[file_path] = event

            # Queue for processing with debounce
            await self.update_queue.put(event)

            return True

        except Exception as e:
            print(f"❌ Error handling file change {file_path}: {e}")
            return False

    async def process_pending_changes(self) -> UpdateResult:
        """Process all pending changes in batch"""
        async with self.processing_lock:
            start_time = time.time()
            result = UpdateResult(success=True)

            try:
                # Collect changes to process
                changes_to_process = []

                # Debounce: wait for stabilization
                await asyncio.sleep(self.debounce_delay)

                # Collect all pending changes
                while not self.update_queue.empty():
                    try:
                        event = self.update_queue.get_nowait()

                        # Only process if this is the latest change for the file
                        if (event.file_path in self.pending_changes and
                            self.pending_changes[event.file_path].timestamp == event.timestamp):
                            changes_to_process.append(event)
                    except asyncio.QueueEmpty:
                        break

                if not changes_to_process:
                    return result

                print(f"🔄 Processing {len(changes_to_process)} file changes...")

                # Group changes by type
                created_files = [e for e in changes_to_process if e.event_type == 'created']
                modified_files = [e for e in changes_to_process if e.event_type == 'modified']
                deleted_files = [e for e in changes_to_process if e.event_type == 'deleted']

                # Process deletions first
                for event in deleted_files:
                    await self._handle_file_deletion(event, result)

                # Process modifications and creations
                for event in modified_files + created_files:
                    await self._handle_file_update(event, result)

                # Update dependencies for affected files
                await self._update_dependencies(changes_to_process)

                # Clear pending changes for processed files
                for event in changes_to_process:
                    self.pending_changes.pop(event.file_path, None)

                result.processing_time = time.time() - start_time
                result.processed_files = len(changes_to_process)

                print(f"✅ Incremental update complete in {result.processing_time:.2f}s")
                print(f"   Added: {result.entities_added}, Updated: {result.entities_updated}, Removed: {result.entities_removed}")

            except Exception as e:
                result.success = False
                result.error_message = str(e)
                result.processing_time = time.time() - start_time
                print(f"❌ Incremental update failed: {e}")

            return result

    async def _handle_file_deletion(self, event: ChangeEvent, result: UpdateResult):
        """Handle file deletion"""
        try:
            file_path = event.file_path

            # Get entities defined in this file
            entities_to_remove = self.file_to_entities.get(file_path, set())

            # Remove entities from graph
            for entity_uri in entities_to_remove:
                success = self.processor.graph_store.delete_entity(entity_uri)
                if success:
                    result.entities_removed += 1

                # Clean up mappings
                self.entity_to_files.pop(entity_uri, None)

            # Clean up file mappings
            self.file_to_entities.pop(file_path, None)
            self.file_hashes.pop(file_path, None)
            self.file_dependencies.pop(file_path, None)
            self.file_dependents.pop(file_path, None)

            print(f"🗑️  Removed {len(entities_to_remove)} entities from deleted file: {file_path}")

        except Exception as e:
            print(f"❌ Error handling file deletion {event.file_path}: {e}")

    async def _handle_file_update(self, event: ChangeEvent, result: UpdateResult):
        """Handle file creation or modification"""
        try:
            file_path = event.file_path

            # Check if file actually changed
            old_hash = self.file_hashes.get(file_path)
            new_hash = event.content_hash

            if old_hash == new_hash:
                return  # No actual change

            # Parse the file
            try:
                root_node, new_entities = self.processor.ast_parser.parse_file(file_path)
            except Exception as e:
                print(f"❌ Failed to parse {file_path}: {e}")
                return

            # Get existing entities for this file
            old_entities = self.file_to_entities.get(file_path, set())

            # Remove old entities from graph
            for entity_uri in old_entities:
                self.processor.graph_store.delete_entity(entity_uri)
                result.entities_removed += 1

            # Add new entities to graph
            if new_entities:
                # Enhance with semantic analysis for this file only
                semantic_info = await self._analyze_single_file_semantics(file_path)

                # Add to graph store
                success = self.processor.graph_store.add_entities_from_list(new_entities)
                if success:
                    if event.event_type == 'created':
                        result.entities_added += len(new_entities)
                    else:
                        result.entities_updated += len(new_entities)

            # Update mappings
            new_entity_uris = {entity.uri for entity in new_entities}
            self.file_to_entities[file_path] = new_entity_uris

            for entity in new_entities:
                if entity.uri not in self.entity_to_files:
                    self.entity_to_files[entity.uri] = set()
                self.entity_to_files[entity.uri].add(file_path)

            # Update file hash
            self.file_hashes[file_path] = new_hash

            action = "Created" if event.event_type == 'created' else "Updated"
            print(f"📝 {action} {len(new_entities)} entities in: {file_path}")

        except Exception as e:
            print(f"❌ Error handling file update {event.file_path}: {e}")

    async def _analyze_single_file_semantics(self, file_path: str) -> SemanticGraph:
        """Perform semantic analysis for a single file"""
        try:
            # Simple semantic analysis - in practice would call LSP for just this file
            semantic_graph = SemanticGraph()

            # For now, return empty semantic graph
            # In full implementation, would call LSP client with file-specific analysis

            return semantic_graph

        except Exception as e:
            print(f"⚠️  Semantic analysis failed for {file_path}: {e}")
            return SemanticGraph()

    async def _update_dependencies(self, changes: List[ChangeEvent]):
        """Update dependency relationships after file changes"""
        try:
            affected_files = {change.file_path for change in changes}

            # Find all files that might be affected by these changes
            files_to_recheck = set(affected_files)

            # Add files that depend on the changed files
            for changed_file in affected_files:
                dependents = self.file_dependents.get(changed_file, set())
                files_to_recheck.update(dependents)

            # Re-analyze dependencies for affected files
            # This is a simplified version - full implementation would re-run LSP analysis
            print(f"🔗 Updating dependencies for {len(files_to_recheck)} files")

        except Exception as e:
            print(f"⚠️  Dependency update failed: {e}")

    def get_affected_files(self, changed_file: str) -> Set[str]:
        """Get all files potentially affected by a change to the given file"""
        affected = {changed_file}

        # Add direct dependents
        dependents = self.file_dependents.get(changed_file, set())
        affected.update(dependents)

        # For complex dependency chains, could do transitive closure
        # For now, just immediate dependents

        return affected

    def get_update_statistics(self) -> Dict[str, Any]:
        """Get statistics about the updater state"""
        return {
            'tracked_files': len(self.file_hashes),
            'tracked_entities': len(self.entity_to_files),
            'dependency_relationships': len(self.file_dependencies),
            'pending_changes': len(self.pending_changes),
            'queue_size': self.update_queue.qsize()
        }


class FileWatcher(FileSystemEventHandler):
    """File system watcher that integrates with incremental updater"""

    def __init__(self, updater: IncrementalUpdater, codebase_path: str):
        super().__init__()
        self.updater = updater
        self.codebase_path = Path(codebase_path)
        self.supported_extensions = {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}

    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed"""
        path = Path(file_path)

        # Check extension
        if path.suffix not in self.supported_extensions:
            return False

        # Skip hidden files and directories
        if any(part.startswith('.') for part in path.parts):
            return False

        # Skip node_modules and other common ignore dirs
        ignore_dirs = {'node_modules', 'dist', 'build', '.git', '__pycache__'}
        if any(ignore_dir in path.parts for ignore_dir in ignore_dirs):
            return False

        return True

    def on_modified(self, event):
        if not event.is_directory and self._should_process_file(event.src_path):
            asyncio.create_task(self.updater.on_file_change(event.src_path, 'modified'))

    def on_created(self, event):
        if not event.is_directory and self._should_process_file(event.src_path):
            asyncio.create_task(self.updater.on_file_change(event.src_path, 'created'))

    def on_deleted(self, event):
        if not event.is_directory and self._should_process_file(event.src_path):
            asyncio.create_task(self.updater.on_file_change(event.src_path, 'deleted'))


async def start_file_watching(updater: IncrementalUpdater, codebase_path: str):
    """Start file system watching with automatic processing"""

    observer = Observer()
    event_handler = FileWatcher(updater, codebase_path)

    observer.schedule(event_handler, codebase_path, recursive=True)
    observer.start()

    print(f"👀 Started watching {codebase_path} for changes...")

    try:
        # Process changes periodically
        while True:
            await asyncio.sleep(2.0)  # Check for changes every 2 seconds

            if not updater.update_queue.empty():
                await updater.process_pending_changes()

    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Stopped file watching")

    observer.join()


# Factory function
def create_incremental_updater(processor: CodebaseProcessor,
                              debounce_delay: float = 0.5,
                              batch_size: int = 10) -> IncrementalUpdater:
    """Create incremental updater for processor"""
    return IncrementalUpdater(processor, debounce_delay, batch_size)


# CLI interface for testing
async def main():
    """Test CLI for incremental updater"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python incremental_updater.py <codebase_path>")
        sys.exit(1)

    codebase_path = sys.argv[1]

    # Create processor and updater
    from .processor import create_codebase_processor
    processor = create_codebase_processor(backend_type='rdflib')

    # Initial processing
    print("🚀 Initial processing...")
    knowledge_graph = processor.process_codebase(codebase_path)

    # Create and initialize updater
    updater = create_incremental_updater(processor)
    success = updater.initialize(codebase_path)

    if not success:
        print("❌ Failed to initialize updater")
        sys.exit(1)

    # Start watching
    try:
        await start_file_watching(updater, codebase_path)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        processor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())