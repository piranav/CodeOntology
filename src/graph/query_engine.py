"""
Advanced query engine for code knowledge graphs.
Provides high-level query interface with pattern matching and context retrieval.
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .graph_store import GraphStore


class QueryType(Enum):
    """Types of supported queries"""
    SPARQL = "sparql"
    PATTERN = "pattern"
    NATURAL = "natural"
    GRAPH_TRAVERSAL = "traversal"


@dataclass
class QueryResult:
    """Standardized query result format"""
    query_type: QueryType
    results: List[Dict[str, Any]]
    execution_time: float
    total_results: int
    query: str
    context: Optional[Dict[str, Any]] = None
    success: bool = True


@dataclass
class GraphPattern:
    """Graph pattern for pattern matching queries"""
    nodes: List[Dict[str, Any]]  # Node constraints
    edges: List[Dict[str, Any]]  # Edge constraints
    filters: Dict[str, Any]  # Additional filters


class QueryEngine:
    """
    High-level query engine providing multiple query interfaces
    This is where the value proposition lives - precise, context-aware queries
    """

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store
        self.backend_type = graph_store.backend_type

        # Pre-defined query templates for common patterns
        self.query_templates = self._initialize_query_templates()

        # Query cache for performance
        self._query_cache: Dict[str, QueryResult] = {}
        self._cache_size_limit = 100

    def _initialize_query_templates(self) -> Dict[str, str]:
        """Initialize common SPARQL query templates"""
        return {
            'functions_calling_function': """
            SELECT ?caller ?callerName WHERE {{
                ?caller code:calls <{target_function}> .
                ?caller code:hasName ?callerName .
            }}
            """,

            'functions_called_by_function': """
            SELECT ?callee ?calleeName WHERE {{
                <{source_function}> code:calls ?callee .
                ?callee code:hasName ?calleeName .
            }}
            """,

            'classes_implementing_interface': """
            SELECT ?class ?className WHERE {{
                ?class code:implements <{interface}> .
                ?class code:hasName ?className .
            }}
            """,

            'functions_in_module': """
            SELECT ?function ?functionName WHERE {{
                <{module}> code:defines ?function .
                ?function a code:Function .
                ?function code:hasName ?functionName .
            }}
            """,

            'variables_accessed_by_function': """
            SELECT ?variable ?varName WHERE {{
                <{function}> code:references ?variable .
                ?variable a code:Variable .
                ?variable code:hasName ?varName .
            }}
            """,

            'call_chain_between_functions': """
            SELECT ?intermediate WHERE {{
                <{source_function}> code:calls+ ?intermediate .
                ?intermediate code:calls+ <{target_function}> .
            }}
            """,

            'circular_dependencies': """
            SELECT ?module1 ?module2 WHERE {{
                ?module1 code:imports ?module2 .
                ?module2 code:imports+ ?module1 .
            }}
            """,

            'database_operations': """
            SELECT ?function ?functionName WHERE {{
                ?function code:calls ?dbCall .
                ?dbCall code:hasName ?dbCallName .
                ?function code:hasName ?functionName .
                FILTER(
                    CONTAINS(LCASE(?dbCallName), "query") ||
                    CONTAINS(LCASE(?dbCallName), "select") ||
                    CONTAINS(LCASE(?dbCallName), "insert") ||
                    CONTAINS(LCASE(?dbCallName), "update") ||
                    CONTAINS(LCASE(?dbCallName), "delete") ||
                    CONTAINS(LCASE(?dbCallName), "find") ||
                    CONTAINS(LCASE(?dbCallName), "save")
                )
            }}
            """,

            'unused_functions': """
            SELECT ?function ?functionName WHERE {{
                ?function a code:Function .
                ?function code:hasName ?functionName .
                ?function code:isExported false .
                MINUS {{
                    ?caller code:calls ?function .
                }}
            }}
            """,

            'high_complexity_functions': """
            SELECT ?function ?functionName (COUNT(?call) as ?callCount) WHERE {{
                ?function a code:Function .
                ?function code:hasName ?functionName .
                ?function code:calls ?call .
            }}
            GROUP BY ?function ?functionName
            HAVING (?callCount > {complexity_threshold})
            ORDER BY DESC(?callCount)
            """,

            'functions_with_many_parameters': """
            SELECT ?function ?functionName (COUNT(?param) as ?paramCount) WHERE {{
                ?function a code:Function .
                ?function code:hasName ?functionName .
                ?function code:hasParameter ?param .
            }}
            GROUP BY ?function ?functionName
            HAVING (?paramCount > {param_threshold})
            ORDER BY DESC(?paramCount)
            """
        }

    def sparql_query(self, query: str, use_cache: bool = True) -> QueryResult:
        """Execute raw SPARQL query"""
        import time

        # Check cache first
        cache_key = f"sparql:{hash(query)}"
        if use_cache and cache_key in self._query_cache:
            cached_result = self._query_cache[cache_key]
            print(f"Query cache hit: {len(cached_result.results)} results")
            return cached_result

        start_time = time.time()

        try:
            results = self.graph_store.query(query)
            execution_time = time.time() - start_time

            result = QueryResult(
                query_type=QueryType.SPARQL,
                results=results,
                execution_time=execution_time,
                total_results=len(results),
                query=query
            )

            # Cache result if beneficial
            if use_cache and len(results) < 1000:  # Don't cache huge results
                self._add_to_cache(cache_key, result)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"SPARQL query failed: {e}")

            return QueryResult(
                query_type=QueryType.SPARQL,
                results=[],
                execution_time=execution_time,
                total_results=0,
                query=query,
                context={'error': str(e)}
            )

    def pattern_match(self, pattern: GraphPattern) -> QueryResult:
        """Find subgraph matches using pattern matching"""
        import time

        start_time = time.time()
        results = []

        try:
            if self.backend_type == 'networkx':
                # Use NetworkX for graph pattern matching
                results = self._networkx_pattern_match(pattern)
            else:
                # Convert pattern to SPARQL
                sparql_query = self._pattern_to_sparql(pattern)
                sparql_result = self.sparql_query(sparql_query, use_cache=False)
                results = sparql_result.results

            execution_time = time.time() - start_time

            return QueryResult(
                query_type=QueryType.PATTERN,
                results=results,
                execution_time=execution_time,
                total_results=len(results),
                query=str(pattern)
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Pattern matching failed: {e}")

            return QueryResult(
                query_type=QueryType.PATTERN,
                results=[],
                execution_time=execution_time,
                total_results=0,
                query=str(pattern),
                context={'error': str(e)}
            )

    def get_context(self, entity_uri: str, depth: int = 2,
                   relationship_types: Optional[List[str]] = None) -> QueryResult:
        """
        Get comprehensive context around an entity
        This replaces embedding-based retrieval with precise graph traversal
        """
        import time

        start_time = time.time()

        try:
            # Get direct context from graph store
            context_data = self.graph_store.get_context(entity_uri, depth)

            # Enhance with relationship type filtering
            if relationship_types:
                filtered_relationships = [
                    rel for rel in context_data.get('relationships', [])
                    if rel.get('relationship') in relationship_types
                ]
                context_data['relationships'] = filtered_relationships

            # Add semantic enrichment
            enriched_context = self._enrich_context(entity_uri, context_data)

            execution_time = time.time() - start_time

            return QueryResult(
                query_type=QueryType.GRAPH_TRAVERSAL,
                results=[enriched_context],
                execution_time=execution_time,
                total_results=1,
                query=f"context({entity_uri}, depth={depth})",
                context={'entity_uri': entity_uri, 'depth': depth}
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"Context retrieval failed: {e}")

            return QueryResult(
                query_type=QueryType.GRAPH_TRAVERSAL,
                results=[],
                execution_time=execution_time,
                total_results=0,
                query=f"context({entity_uri}, depth={depth})",
                context={'error': str(e), 'entity_uri': entity_uri}
            )

    def natural_language_query(self, nl_query: str) -> QueryResult:
        """
        Convert natural language to graph query (basic implementation)
        In production, this would use LLM to generate SPARQL
        """
        import time

        start_time = time.time()

        # Simple pattern matching for common queries
        query_patterns = {
            r"find.*functions.*call.*database": 'database_operations',
            r"find.*functions.*call.*(\w+)": lambda m: self._find_functions_calling(m.group(1)),
            r"classes.*implement.*(\w+)": lambda m: self._find_classes_implementing(m.group(1)),
            r"unused.*functions": 'unused_functions',
            r"circular.*dependencies": 'circular_dependencies',
            r"complex.*functions": 'high_complexity_functions',
            r"functions.*many.*parameters": 'functions_with_many_parameters'
        }

        nl_query_lower = nl_query.lower()
        results = []
        matched_template = None

        for pattern, template_or_func in query_patterns.items():
            match = re.search(pattern, nl_query_lower)
            if match:
                if callable(template_or_func):
                    sparql_query = template_or_func(match)
                else:
                    template = self.query_templates.get(template_or_func)
                    if template:
                        # Fill in default parameters
                        sparql_query = template.format(
                            complexity_threshold=5,
                            param_threshold=4
                        )
                    else:
                        continue

                sparql_result = self.sparql_query(sparql_query, use_cache=False)
                results = sparql_result.results
                matched_template = template_or_func
                break

        execution_time = time.time() - start_time

        return QueryResult(
            query_type=QueryType.NATURAL,
            results=results,
            execution_time=execution_time,
            total_results=len(results),
            query=nl_query,
            context={
                'matched_template': matched_template,
                'parsed_query': sparql_query if 'sparql_query' in locals() else None
            }
        )

    def _find_functions_calling(self, function_name: str) -> str:
        """Generate SPARQL to find functions calling a specific function"""
        return f"""
        SELECT ?caller ?callerName WHERE {{
            ?caller code:calls ?target .
            ?target code:hasName "{function_name}" .
            ?caller code:hasName ?callerName .
        }}
        """

    def _find_classes_implementing(self, interface_name: str) -> str:
        """Generate SPARQL to find classes implementing an interface"""
        return f"""
        SELECT ?class ?className WHERE {{
            ?class code:implements ?interface .
            ?interface code:hasName "{interface_name}" .
            ?class code:hasName ?className .
        }}
        """

    def _networkx_pattern_match(self, pattern: GraphPattern) -> List[Dict[str, Any]]:
        """Pattern matching using NetworkX"""
        results = []

        try:
            # Simple implementation - would need more sophisticated graph matching
            graph = self.graph_store.backend.graph

            # Find nodes matching pattern constraints
            candidate_nodes = []
            for node, data in graph.nodes(data=True):
                matches = True
                for constraint in pattern.nodes:
                    for key, value in constraint.items():
                        if key not in data or data[key] != value:
                            matches = False
                            break
                    if not matches:
                        break

                if matches:
                    candidate_nodes.append(node)

            # For each candidate node, check edge patterns
            for node in candidate_nodes:
                node_matches = {'root_node': node}
                pattern_satisfied = True

                for edge_constraint in pattern.edges:
                    source_type = edge_constraint.get('source_type')
                    target_type = edge_constraint.get('target_type')
                    edge_type = edge_constraint.get('edge_type')

                    # Find matching edges
                    found_edge = False
                    for neighbor in graph.neighbors(node):
                        edge_data = graph.get_edge_data(node, neighbor)
                        if edge_data and edge_type in edge_data:
                            # Check if neighbor matches target constraint
                            neighbor_data = graph.nodes[neighbor]
                            if target_type and neighbor_data.get('type') == target_type:
                                node_matches[f'target_{target_type}'] = neighbor
                                found_edge = True
                                break

                    if not found_edge:
                        pattern_satisfied = False
                        break

                if pattern_satisfied:
                    results.append(node_matches)

        except Exception as e:
            print(f"NetworkX pattern matching failed: {e}")

        return results

    def _pattern_to_sparql(self, pattern: GraphPattern) -> str:
        """Convert graph pattern to SPARQL query"""
        # This is a simplified conversion
        select_vars = []
        where_clauses = []
        filters = []

        # Process nodes
        for i, node_constraint in enumerate(pattern.nodes):
            var_name = f"node{i}"
            select_vars.append(f"?{var_name}")

            node_type = node_constraint.get('type')
            if node_type:
                where_clauses.append(f"?{var_name} a code:{node_type} .")

            for prop, value in node_constraint.items():
                if prop != 'type':
                    where_clauses.append(f"?{var_name} code:{prop} \"{value}\" .")

        # Process edges
        for edge_constraint in pattern.edges:
            source_idx = edge_constraint.get('source_index', 0)
            target_idx = edge_constraint.get('target_index', 1)
            relationship = edge_constraint.get('relationship', 'related')

            source_var = f"node{source_idx}"
            target_var = f"node{target_idx}"

            where_clauses.append(f"?{source_var} code:{relationship} ?{target_var} .")

        # Process filters
        for filter_key, filter_value in pattern.filters.items():
            if isinstance(filter_value, str) and '*' in filter_value:
                # Wildcard filter
                filter_pattern = filter_value.replace('*', '.*')
                filters.append(f"REGEX(?{filter_key}, \"{filter_pattern}\", \"i\")")
            else:
                filters.append(f"?{filter_key} = \"{filter_value}\"")

        # Construct SPARQL
        select_clause = "SELECT " + " ".join(select_vars) if select_vars else "SELECT *"
        where_clause = "WHERE {\n  " + "\n  ".join(where_clauses)

        if filters:
            filter_clause = "\n  FILTER(" + " && ".join(filters) + ")"
            where_clause += filter_clause

        where_clause += "\n}"

        return f"{select_clause}\n{where_clause}"

    def _enrich_context(self, entity_uri: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add semantic enrichment to context"""
        enriched = context_data.copy()

        # Add entity metadata
        try:
            if self.backend_type == 'rdflib':
                # Get entity properties
                entity_query = f"""
                SELECT ?property ?value WHERE {{
                    <{entity_uri}> ?property ?value .
                }}
                """
                properties = self.graph_store.query(entity_query)
                enriched['entity_properties'] = properties

                # Get type hierarchy
                type_query = f"""
                SELECT ?type WHERE {{
                    <{entity_uri}> a ?type .
                }}
                """
                types = self.graph_store.query(type_query)
                enriched['entity_types'] = [t['type'] for t in types if t['type']]

        except Exception as e:
            print(f"Context enrichment failed: {e}")

        return enriched

    def _add_to_cache(self, cache_key: str, result: QueryResult):
        """Add result to query cache with LRU eviction"""
        if len(self._query_cache) >= self._cache_size_limit:
            # Remove oldest entry (simple FIFO, could be improved to LRU)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]

        self._query_cache[cache_key] = result

    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Provide query suggestions based on partial input"""
        suggestions = []

        # Template-based suggestions
        for template_name in self.query_templates.keys():
            readable_name = template_name.replace('_', ' ').title()
            if partial_query.lower() in readable_name.lower():
                suggestions.append(readable_name)

        # Natural language suggestions
        nl_patterns = [
            "Find functions that call database operations",
            "Find classes implementing interface",
            "Find unused functions",
            "Find circular dependencies",
            "Find complex functions",
            "Find functions with many parameters"
        ]

        for pattern in nl_patterns:
            if partial_query.lower() in pattern.lower():
                suggestions.append(pattern)

        return suggestions[:10]  # Limit to top 10

    def explain_query(self, query: str) -> Dict[str, Any]:
        """Explain query execution plan and performance characteristics"""
        explanation = {
            'query': query,
            'estimated_complexity': 'unknown',
            'suggested_optimizations': [],
            'expected_result_types': []
        }

        try:
            # Analyze SPARQL query structure
            if 'SELECT' in query.upper():
                # Count triple patterns
                triple_count = query.count('?')
                if triple_count > 10:
                    explanation['estimated_complexity'] = 'high'
                    explanation['suggested_optimizations'].append(
                        'Consider adding more specific constraints to reduce search space'
                    )
                elif triple_count > 5:
                    explanation['estimated_complexity'] = 'medium'
                else:
                    explanation['estimated_complexity'] = 'low'

                # Detect potentially expensive operations
                if 'code:calls+' in query or 'code:calls*' in query:
                    explanation['suggested_optimizations'].append(
                        'Transitive closure queries can be expensive on large graphs'
                    )

                if 'REGEX' in query.upper():
                    explanation['suggested_optimizations'].append(
                        'Regular expressions can be slow - consider exact matches if possible'
                    )

        except Exception as e:
            explanation['analysis_error'] = str(e)

        return explanation

    def benchmark_queries(self, queries: List[str], iterations: int = 3) -> Dict[str, Any]:
        """Benchmark query performance"""
        results = {}

        for query in queries:
            times = []
            for _ in range(iterations):
                result = self.sparql_query(query, use_cache=False)
                times.append(result.execution_time)

            results[query[:50] + '...' if len(query) > 50 else query] = {
                'average_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'last_result_count': result.total_results
            }

        return results


# Factory function
def create_query_engine(graph_store: GraphStore) -> QueryEngine:
    """Create query engine for graph store"""
    return QueryEngine(graph_store)
