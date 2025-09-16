"""
Graph storage layer supporting multiple backends.
Provides unified interface for RDF storage, querying, and updates.
"""

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
import networkx as nx
import rdflib
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

from ..models.code_entities import CodeEntity


class GraphStorageBackend(ABC):
    """Abstract base class for graph storage backends"""

    @abstractmethod
    def save(self, graph_data: Any, filepath: str) -> bool:
        """Save graph to storage"""
        pass

    @abstractmethod
    def load(self, filepath: str) -> Any:
        """Load graph from storage"""
        pass

    @abstractmethod
    def query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute query against graph"""
        pass

    @abstractmethod
    def update(self, entity_uri: str, changes: Dict[str, Any]) -> bool:
        """Update entity in graph"""
        pass

    @abstractmethod
    def delete(self, entity_uri: str) -> bool:
        """Delete entity from graph"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        pass


class RDFLibBackend(GraphStorageBackend):
    """RDFLib backend for standard RDF compliance"""

    def __init__(self):
        self.graph = Graph()
        self.CODE = Namespace("http://codeontology.org/")
        self.CODEBASE = Namespace("http://codebase.local/")

        # Bind namespaces
        self.graph.bind("code", self.CODE)
        self.graph.bind("codebase", self.CODEBASE)

    def save(self, graph_data: Graph, filepath: str) -> bool:
        """Save RDF graph to file"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Determine format from extension
            if path.suffix == '.ttl':
                format_type = 'turtle'
            elif path.suffix == '.xml':
                format_type = 'xml'
            elif path.suffix == '.n3':
                format_type = 'n3'
            elif path.suffix == '.nt':
                format_type = 'nt'
            else:
                format_type = 'turtle'  # Default

            graph_data.serialize(destination=filepath, format=format_type)

            # Save metadata
            metadata = {
                'created': datetime.now().isoformat(),
                'format': format_type,
                'triples_count': len(graph_data),
                'namespaces': dict(graph_data.namespaces())
            }

            metadata_path = path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            print(f"Failed to save RDF graph: {e}")
            return False

    def load(self, filepath: str) -> Graph:
        """Load RDF graph from file"""
        try:
            path = Path(filepath)
            if not path.exists():
                print(f"Graph file not found: {filepath}")
                return Graph()

            graph = Graph()

            # Determine format from extension
            if path.suffix == '.ttl':
                format_type = 'turtle'
            elif path.suffix == '.xml':
                format_type = 'xml'
            elif path.suffix == '.n3':
                format_type = 'n3'
            elif path.suffix == '.nt':
                format_type = 'nt'
            else:
                format_type = 'turtle'

            graph.parse(filepath, format=format_type)
            self.graph = graph

            # Load metadata if available
            metadata_path = path.with_suffix('.metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    print(f"Loaded graph with {metadata.get('triples_count', 0)} triples")

            return graph

        except Exception as e:
            print(f"Failed to load RDF graph: {e}")
            return Graph()

    def query(self, sparql_query: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute SPARQL query"""
        try:
            results = []
            query_result = self.graph.query(sparql_query)

            for row in query_result:
                result_row = {}
                if query_result.vars:
                    for var_name in query_result.vars:
                        value = row[var_name] if row[var_name] else None
                        result_row[str(var_name)] = str(value) if value else None
                else:
                    # Boolean result
                    result_row['result'] = bool(row)
                results.append(result_row)

            return results

        except Exception as e:
            print(f"SPARQL query failed: {e}")
            return []

    def update(self, entity_uri: str, changes: Dict[str, Any]) -> bool:
        """Update entity properties"""
        try:
            entity = URIRef(entity_uri)

            # Remove old triples for properties being updated
            for property_name, new_value in changes.items():
                predicate = self.CODE[property_name]

                # Remove existing triples
                triples_to_remove = list(self.graph.triples((entity, predicate, None)))
                for triple in triples_to_remove:
                    self.graph.remove(triple)

                # Add new value
                if isinstance(new_value, str):
                    self.graph.add((entity, predicate, Literal(new_value)))
                elif isinstance(new_value, bool):
                    self.graph.add((entity, predicate, Literal(new_value, datatype=XSD.boolean)))
                elif isinstance(new_value, int):
                    self.graph.add((entity, predicate, Literal(new_value, datatype=XSD.integer)))
                elif isinstance(new_value, list):
                    # Handle list properties
                    for item in new_value:
                        self.graph.add((entity, predicate, URIRef(item) if item.startswith('http') else Literal(item)))

            return True

        except Exception as e:
            print(f"Failed to update entity: {e}")
            return False

    def delete(self, entity_uri: str) -> bool:
        """Delete entity and all its triples"""
        try:
            entity = URIRef(entity_uri)

            # Remove all triples where entity is subject
            triples_to_remove = list(self.graph.triples((entity, None, None)))
            for triple in triples_to_remove:
                self.graph.remove(triple)

            # Remove all triples where entity is object
            triples_to_remove = list(self.graph.triples((None, None, entity)))
            for triple in triples_to_remove:
                self.graph.remove(triple)

            return True

        except Exception as e:
            print(f"Failed to delete entity: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            'total_triples': len(self.graph),
            'namespaces': dict(self.graph.namespaces()),
            'entity_counts': {}
        }

        # Count entities by type
        type_query = """
        SELECT ?type (COUNT(?entity) as ?count) WHERE {
            ?entity a ?type .
        } GROUP BY ?type
        """

        try:
            results = self.query(type_query)
            for result in results:
                if result['type']:
                    type_name = result['type'].split('/')[-1]  # Extract class name
                    stats['entity_counts'][type_name] = int(result['count']) if result['count'] else 0
        except:
            pass

        return stats


class NetworkXBackend(GraphStorageBackend):
    """NetworkX backend for graph analysis and visualization"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed multigraph to handle multiple edge types
        self.entity_data: Dict[str, Dict[str, Any]] = {}  # Store entity attributes

    def save(self, graph_data: nx.MultiDiGraph, filepath: str) -> bool:
        """Save NetworkX graph"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save as GraphML (preserves attributes)
            if path.suffix == '.graphml':
                nx.write_graphml(graph_data, filepath)
            elif path.suffix == '.gexf':
                nx.write_gexf(graph_data, filepath)
            elif path.suffix == '.pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        'graph': graph_data,
                        'entity_data': self.entity_data
                    }, f)
            else:
                # Default to pickle
                with open(filepath + '.pickle', 'wb') as f:
                    pickle.dump({
                        'graph': graph_data,
                        'entity_data': self.entity_data
                    }, f)

            return True

        except Exception as e:
            print(f"Failed to save NetworkX graph: {e}")
            return False

    def load(self, filepath: str) -> nx.MultiDiGraph:
        """Load NetworkX graph"""
        try:
            path = Path(filepath)
            if not path.exists():
                return nx.MultiDiGraph()

            if path.suffix == '.graphml':
                self.graph = nx.read_graphml(filepath)
            elif path.suffix == '.gexf':
                self.graph = nx.read_gexf(filepath)
            elif path.suffix == '.pickle' or not path.suffix:
                pickle_path = filepath if path.suffix else filepath + '.pickle'
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    self.graph = data['graph']
                    self.entity_data = data.get('entity_data', {})

            return self.graph

        except Exception as e:
            print(f"Failed to load NetworkX graph: {e}")
            return nx.MultiDiGraph()

    def query(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute graph queries using NetworkX"""
        results = []

        try:
            if query_type == 'find_nodes_by_type':
                node_type = kwargs.get('node_type')
                for node, data in self.graph.nodes(data=True):
                    if data.get('type') == node_type:
                        results.append({'node': node, **data})

            elif query_type == 'find_neighbors':
                node = kwargs.get('node')
                max_depth = kwargs.get('max_depth', 1)

                if node in self.graph:
                    if max_depth == 1:
                        neighbors = list(self.graph.neighbors(node))
                        for neighbor in neighbors:
                            results.append({
                                'neighbor': neighbor,
                                'relationship': self.graph[node][neighbor]
                            })
                    else:
                        # BFS for multi-hop neighbors
                        visited = set()
                        queue = [(node, 0)]

                        while queue:
                            current, depth = queue.pop(0)
                            if depth >= max_depth:
                                continue

                            for neighbor in self.graph.neighbors(current):
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append((neighbor, depth + 1))
                                    results.append({
                                        'neighbor': neighbor,
                                        'depth': depth + 1,
                                        'path': nx.shortest_path(self.graph, node, neighbor)
                                    })

            elif query_type == 'find_paths':
                source = kwargs.get('source')
                target = kwargs.get('target')
                max_length = kwargs.get('max_length', 10)

                try:
                    paths = list(nx.all_simple_paths(
                        self.graph, source, target, cutoff=max_length
                    ))
                    for path in paths:
                        results.append({'path': path, 'length': len(path) - 1})
                except nx.NetworkXNoPath:
                    pass

            elif query_type == 'connected_components':
                # For directed graphs, use strongly connected components
                components = list(nx.strongly_connected_components(self.graph))
                for i, component in enumerate(components):
                    results.append({
                        'component_id': i,
                        'nodes': list(component),
                        'size': len(component)
                    })

            elif query_type == 'centrality':
                centrality_type = kwargs.get('centrality_type', 'betweenness')

                if centrality_type == 'betweenness':
                    centrality = nx.betweenness_centrality(self.graph)
                elif centrality_type == 'degree':
                    centrality = nx.degree_centrality(self.graph)
                elif centrality_type == 'pagerank':
                    centrality = nx.pagerank(self.graph)
                else:
                    centrality = nx.degree_centrality(self.graph)

                for node, score in centrality.items():
                    results.append({'node': node, 'centrality_score': score})

                # Sort by centrality score
                results.sort(key=lambda x: x['centrality_score'], reverse=True)

        except Exception as e:
            print(f"Graph query failed: {e}")

        return results

    def update(self, entity_uri: str, changes: Dict[str, Any]) -> bool:
        """Update node attributes"""
        try:
            if entity_uri in self.graph:
                # Update node attributes
                for key, value in changes.items():
                    self.graph.nodes[entity_uri][key] = value

                # Update entity data
                if entity_uri not in self.entity_data:
                    self.entity_data[entity_uri] = {}
                self.entity_data[entity_uri].update(changes)

                return True

        except Exception as e:
            print(f"Failed to update node: {e}")

        return False

    def delete(self, entity_uri: str) -> bool:
        """Delete node and its edges"""
        try:
            if entity_uri in self.graph:
                self.graph.remove_node(entity_uri)

            if entity_uri in self.entity_data:
                del self.entity_data[entity_uri]

            return True

        except Exception as e:
            print(f"Failed to delete node: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'is_directed': self.graph.is_directed(),
            'density': nx.density(self.graph),
        }

        # Calculate additional metrics if graph is not empty
        if stats['nodes'] > 0:
            try:
                stats['average_clustering'] = nx.average_clustering(self.graph)
            except:
                stats['average_clustering'] = 0

            # Count nodes by type
            type_counts = {}
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            stats['node_types'] = type_counts

        return stats


class GraphStore:
    """
    Unified graph storage interface supporting multiple backends
    Automatically chooses optimal backend based on use case
    """

    def __init__(self, backend_type: str = 'rdflib', storage_path: Optional[str] = None):
        self.backend_type = backend_type
        self.storage_path = storage_path or 'graph_data'

        # Initialize backend
        if backend_type == 'rdflib':
            self.backend = RDFLibBackend()
        elif backend_type == 'networkx':
            self.backend = NetworkXBackend()
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        # Create storage directory
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

    def save(self, format_type: str = "turtle") -> bool:
        """Save graph with incremental backup"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = Path(self.storage_path)

            if self.backend_type == 'rdflib':
                # Map format to file extension
                ext = 'ttl' if format_type == 'turtle' else (
                    'xml' if format_type == 'xml' else (
                        'n3' if format_type == 'n3' else (
                            'nt' if format_type == 'nt' else format_type)))
                filename = f"knowledge_graph_{timestamp}.{ext}"
                filepath = base_path / filename

                # Save current state
                success = self.backend.save(self.backend.graph, str(filepath))

                if success:
                    # Create symlink to latest
                    latest_link = base_path / f"latest.{ext}"
                    if latest_link.exists():
                        latest_link.unlink()
                    latest_link.symlink_to(filename)

            else:  # NetworkX
                filename = f"knowledge_graph_{timestamp}.pickle"
                filepath = base_path / filename
                success = self.backend.save(self.backend.graph, str(filepath))

                if success:
                    latest_link = base_path / "latest.pickle"
                    if latest_link.exists():
                        latest_link.unlink()
                    latest_link.symlink_to(filename)

            return success

        except Exception as e:
            print(f"Failed to save graph: {e}")
            return False

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load graph from file or latest backup"""
        try:
            if not filepath:
                # Load latest
                if self.backend_type == 'rdflib':
                    latest_path = Path(self.storage_path) / "latest.ttl"
                else:
                    latest_path = Path(self.storage_path) / "latest.pickle"

                if latest_path.exists():
                    filepath = str(latest_path)
                else:
                    print("No saved graph found")
                    return False

            self.backend.load(filepath)
            return True

        except Exception as e:
            print(f"Failed to load graph: {e}")
            return False

    def query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute query using appropriate backend"""
        return self.backend.query(query, **kwargs)

    def update_entity(self, entity_uri: str, changes: Dict[str, Any]) -> bool:
        """Update entity with surgical precision"""
        return self.backend.update(entity_uri, changes)

    def delete_entity(self, entity_uri: str) -> bool:
        """Delete entity and clean up references"""
        return self.backend.delete(entity_uri)

    def add_entities_from_list(self, entities: List[CodeEntity]) -> bool:
        """Add multiple entities efficiently"""
        try:
            if self.backend_type == 'rdflib':
                # Use ontology builder to populate RDF graph
                from .ontology_builder import OntologyBuilder
                builder = OntologyBuilder()
                populated_graph = builder.add_entities(entities)
                self.backend.graph = populated_graph

            elif self.backend_type == 'networkx':
                # Add entities as nodes with attributes
                for entity in entities:
                    self.backend.graph.add_node(
                        entity.uri,
                        name=entity.name,
                        type=entity.__class__.__name__,
                        location=entity.location.file_path,
                        line_number=entity.location.line_number,
                        **{k: v for k, v in entity.__dict__.items()
                           if not k.startswith('_') and not callable(v)}
                    )

                    # Add relationships as edges
                    self._add_entity_relationships_networkx(entity)

            return True

        except Exception as e:
            print(f"Failed to add entities: {e}")
            return False

    def _add_entity_relationships_networkx(self, entity: CodeEntity):
        """Add entity relationships as edges in NetworkX"""
        try:
            # Function calls
            if hasattr(entity, 'calls'):
                for called_uri in entity.calls:
                    self.backend.graph.add_edge(
                        entity.uri, called_uri,
                        relationship='calls',
                        type='function_call'
                    )

            # Class inheritance
            if hasattr(entity, 'extends_class') and entity.extends_class:
                self.backend.graph.add_edge(
                    entity.uri, entity.extends_class,
                    relationship='extends',
                    type='inheritance'
                )

            # Interface implementation
            if hasattr(entity, 'implements_interfaces'):
                for interface_uri in entity.implements_interfaces:
                    self.backend.graph.add_edge(
                        entity.uri, interface_uri,
                        relationship='implements',
                        type='interface_implementation'
                    )

            # Variable references
            if hasattr(entity, 'accesses_variables'):
                for var_uri in entity.accesses_variables:
                    self.backend.graph.add_edge(
                        entity.uri, var_uri,
                        relationship='accesses',
                        type='variable_access'
                    )

        except Exception as e:
            print(f"Error adding relationships for {entity.uri}: {e}")

    def get_context(self, entity_uri: str, depth: int = 2) -> Dict[str, Any]:
        """Get contextual information around an entity"""
        context = {
            'entity_uri': entity_uri,
            'depth': depth,
            'related_entities': [],
            'relationships': []
        }

        try:
            if self.backend_type == 'networkx':
                # Use NetworkX graph traversal
                results = self.backend.query('find_neighbors', node=entity_uri, max_depth=depth)
                context['related_entities'] = results

            elif self.backend_type == 'rdflib':
                # Use SPARQL for context query
                context_query = f"""
                SELECT ?related ?relationship WHERE {{
                    {{
                        <{entity_uri}> ?relationship ?related .
                    }} UNION {{
                        ?related ?relationship <{entity_uri}> .
                    }}
                }}
                """
                results = self.backend.query(context_query)
                context['relationships'] = results

        except Exception as e:
            print(f"Failed to get context for {entity_uri}: {e}")

        return context

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        base_stats = self.backend.get_stats()

        # Add storage information
        base_stats['backend_type'] = self.backend_type
        base_stats['storage_path'] = self.storage_path

        # Add file information if available
        storage_path = Path(self.storage_path)
        if storage_path.exists():
            files = list(storage_path.glob('*'))
            base_stats['backup_count'] = len([f for f in files if f.is_file()])

            latest_file = None
            latest_time = 0
            for file in files:
                if file.is_file() and file.stat().st_mtime > latest_time:
                    latest_time = file.stat().st_mtime
                    latest_file = file.name

            base_stats['latest_backup'] = latest_file

        return base_stats

    def export_for_visualization(self, format_type: str = 'json') -> str:
        """Export graph in format suitable for visualization"""
        try:
            if format_type == 'json' and self.backend_type == 'networkx':
                # Export as JSON for D3.js or similar
                from networkx.readwrite import json_graph
                data = json_graph.node_link_data(self.backend.graph)
                return json.dumps(data, indent=2)

            elif format_type == 'graphml':
                # Export as GraphML
                output_path = Path(self.storage_path) / 'visualization.graphml'
                if self.backend_type == 'networkx':
                    nx.write_graphml(self.backend.graph, output_path)
                    return str(output_path)

            elif format_type == 'dot':
                # Export as DOT for Graphviz
                output_path = Path(self.storage_path) / 'visualization.dot'
                if self.backend_type == 'networkx':
                    nx.drawing.nx_agraph.write_dot(self.backend.graph, output_path)
                    return str(output_path)

        except Exception as e:
            print(f"Failed to export for visualization: {e}")

        return ""


# Factory function
def create_graph_store(backend_type: str = 'rdflib', storage_path: Optional[str] = None) -> GraphStore:
    """Create graph store with specified backend"""
    return GraphStore(backend_type, storage_path)
