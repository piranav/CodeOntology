"""
Ontology builder that transforms AST and LSP data into RDF triples.
Populates the knowledge graph according to the code ontology schema.
"""

import rdflib
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, XSD, OWL
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from ..models.code_entities import (
    CodeEntity, FunctionEntity, ClassEntity, VariableEntity,
    ImportEntity, ExportEntity, ModuleEntity, ParameterEntity,
    MethodEntity, PropertyEntity, InterfaceEntity, CallExpressionEntity,
    SourceLocation, TypeInfo
)


class OntologyBuilder:
    """
    Transforms code entities into RDF knowledge graph
    Maintains consistency with the code ontology schema
    """

    def __init__(self, ontology_path: Optional[str] = None):
        # Initialize RDF graph
        self.graph = Graph()

        # Define namespaces
        self.CODE = Namespace("http://codeontology.org/")
        self.CODEBASE = Namespace("http://codebase.local/")
        self.RDF = RDF

        # Bind prefixes
        self.graph.bind("code", self.CODE)
        self.graph.bind("codebase", self.CODEBASE)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
        self.graph.bind("owl", OWL)

        # Load base ontology if provided
        if ontology_path:
            self.load_ontology(ontology_path)

        # Cache for created URIs to avoid duplicates
        self._uri_cache: Set[str] = set()
        self._entity_map: Dict[str, CodeEntity] = {}

    def load_ontology(self, ontology_path: str):
        """Load the base ontology schema"""
        try:
            self.graph.parse(ontology_path, format="turtle")
            print(f"Loaded ontology from {ontology_path}")
        except Exception as e:
            print(f"Failed to load ontology: {e}")

    def add_entities(self, entities: List[CodeEntity]) -> Graph:
        """
        Add multiple entities to the graph
        Returns the populated graph
        """
        # First pass: create all entities
        for entity in entities:
            self._entity_map[entity.uri] = entity
            self._add_entity_to_graph(entity)

        # Second pass: add relationships
        for entity in entities:
            self._add_relationships(entity)

        return self.graph

    def _add_entity_to_graph(self, entity: CodeEntity):
        """Add a single entity to the graph with all its properties"""
        entity_uri = URIRef(entity.uri)

        # Avoid duplicate URIs
        if entity.uri in self._uri_cache:
            return
        self._uri_cache.add(entity.uri)

        # Add basic properties common to all entities
        self._add_basic_properties(entity_uri, entity)

        # Add type-specific properties
        if isinstance(entity, ModuleEntity):
            self._add_module(entity_uri, entity)
        elif isinstance(entity, FunctionEntity):
            self._add_function(entity_uri, entity)
        elif isinstance(entity, MethodEntity):
            self._add_method(entity_uri, entity)
        elif isinstance(entity, ClassEntity):
            self._add_class(entity_uri, entity)
        elif isinstance(entity, InterfaceEntity):
            self._add_interface(entity_uri, entity)
        elif isinstance(entity, VariableEntity):
            self._add_variable(entity_uri, entity)
        elif isinstance(entity, PropertyEntity):
            self._add_property(entity_uri, entity)
        elif isinstance(entity, ParameterEntity):
            self._add_parameter(entity_uri, entity)
        elif isinstance(entity, ImportEntity):
            self._add_import(entity_uri, entity)
        elif isinstance(entity, ExportEntity):
            self._add_export(entity_uri, entity)
        elif isinstance(entity, CallExpressionEntity):
            self._add_call_expression(entity_uri, entity)

    def _add_basic_properties(self, entity_uri: URIRef, entity: CodeEntity):
        """Add properties common to all code entities"""
        # Basic identification
        self.graph.add((entity_uri, self.CODE.hasName, Literal(entity.name, datatype=XSD.string)))
        self.graph.add((entity_uri, self.CODE.hasURI, Literal(entity.uri, datatype=XSD.anyURI)))

        if entity.docstring:
            self.graph.add((entity_uri, self.CODE.hasDocstring, Literal(entity.docstring, datatype=XSD.string)))

        if entity.body_hash:
            self.graph.add((entity_uri, self.CODE.hasBodyHash, Literal(entity.body_hash, datatype=XSD.string)))

        # Add source location
        self._add_source_location(entity_uri, entity.location)

        # Add comments if any
        for comment in entity.comments:
            comment_node = BNode()
            self.graph.add((entity_uri, self.CODE.hasComment, comment_node))
            self.graph.add((comment_node, self.CODE.commentText, Literal(comment, datatype=XSD.string)))

    def _add_source_location(self, entity_uri: URIRef, location: SourceLocation):
        """Add source location information"""
        location_node = BNode()
        self.graph.add((entity_uri, self.CODE.locatedAt, location_node))
        self.graph.add((location_node, RDF.type, self.CODE.SourceLocation))

        self.graph.add((location_node, self.CODE.filePath, Literal(location.file_path, datatype=XSD.string)))
        self.graph.add((location_node, self.CODE.lineNumber, Literal(location.line_number, datatype=XSD.integer)))
        self.graph.add((location_node, self.CODE.columnNumber, Literal(location.column, datatype=XSD.integer)))

        if location.end_line:
            self.graph.add((location_node, self.CODE.endLineNumber, Literal(location.end_line, datatype=XSD.integer)))
        if location.end_column:
            self.graph.add((location_node, self.CODE.endColumnNumber, Literal(location.end_column, datatype=XSD.integer)))

    def _add_module(self, entity_uri: URIRef, entity: ModuleEntity):
        """Add module-specific properties"""
        self.graph.add((entity_uri, RDF.type, self.CODE.Module))

        # Module system information
        self.graph.add((entity_uri, self.CODE.moduleType, Literal(entity.module_type, datatype=XSD.string)))

        # Dependencies
        for dep in entity.dependencies:
            self.graph.add((entity_uri, self.CODE.dependsOn, Literal(dep, datatype=XSD.string)))

    def _add_function(self, entity_uri: URIRef, entity: FunctionEntity):
        """Add function-specific properties"""
        self.graph.add((entity_uri, RDF.type, self.CODE.Function))

        # Function modifiers
        self.graph.add((entity_uri, self.CODE.isAsync, Literal(entity.is_async, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isGenerator, Literal(entity.is_generator, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isArrowFunction, Literal(entity.is_arrow_function, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isExported, Literal(entity.is_exported, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isDefaultExport, Literal(entity.is_default_export, datatype=XSD.boolean)))

        # Scope
        self.graph.add((entity_uri, self.CODE.hasScope, Literal(entity.scope, datatype=XSD.string)))

    def _add_method(self, entity_uri: URIRef, entity: MethodEntity):
        """Add method-specific properties"""
        # Method extends Function
        self._add_function(entity_uri, entity)
        self.graph.add((entity_uri, RDF.type, self.CODE.Method))

        # Method modifiers
        self.graph.add((entity_uri, self.CODE.isStatic, Literal(entity.is_static, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isPrivate, Literal(entity.is_private, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isProtected, Literal(entity.is_protected, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isConstructor, Literal(entity.is_constructor, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isGetter, Literal(entity.is_getter, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isSetter, Literal(entity.is_setter, datatype=XSD.boolean)))

    def _add_class(self, entity_uri: URIRef, entity: ClassEntity):
        """Add class-specific properties"""
        self.graph.add((entity_uri, RDF.type, self.CODE.Class))

        # Class modifiers
        self.graph.add((entity_uri, self.CODE.isAbstract, Literal(entity.is_abstract, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isExported, Literal(entity.is_exported, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isDefaultExport, Literal(entity.is_default_export, datatype=XSD.boolean)))

        # Type parameters
        for type_param in entity.type_parameters:
            self.graph.add((entity_uri, self.CODE.hasTypeParameter, Literal(type_param, datatype=XSD.string)))

    def _add_interface(self, entity_uri: URIRef, entity: InterfaceEntity):
        """Add interface-specific properties"""
        self.graph.add((entity_uri, RDF.type, self.CODE.Interface))

        self.graph.add((entity_uri, self.CODE.isExported, Literal(entity.is_exported, datatype=XSD.boolean)))

        # Type parameters
        for type_param in entity.type_parameters:
            self.graph.add((entity_uri, self.CODE.hasTypeParameter, Literal(type_param, datatype=XSD.string)))

    def _add_variable(self, entity_uri: URIRef, entity: VariableEntity):
        """Add variable-specific properties"""
        self.graph.add((entity_uri, RDF.type, self.CODE.Variable))

        # Variable declaration type
        self.graph.add((entity_uri, self.CODE.isConst, Literal(entity.is_const, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isLet, Literal(entity.is_let, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isVar, Literal(entity.is_var, datatype=XSD.boolean)))

        # Scope and initialization
        self.graph.add((entity_uri, self.CODE.hasScope, Literal(entity.scope, datatype=XSD.string)))

        if entity.initialization_value:
            self.graph.add((entity_uri, self.CODE.initializationValue,
                          Literal(entity.initialization_value, datatype=XSD.string)))

    def _add_property(self, entity_uri: URIRef, entity: PropertyEntity):
        """Add property-specific properties"""
        # Property extends Variable
        self._add_variable(entity_uri, entity)
        self.graph.add((entity_uri, RDF.type, self.CODE.Property))

        # Property modifiers
        self.graph.add((entity_uri, self.CODE.isStatic, Literal(entity.is_static, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isPrivate, Literal(entity.is_private, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isProtected, Literal(entity.is_protected, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isReadonly, Literal(entity.is_readonly, datatype=XSD.boolean)))

    def _add_parameter(self, entity_uri: URIRef, entity: ParameterEntity):
        """Add parameter-specific properties"""
        # Parameter extends Variable
        self.graph.add((entity_uri, RDF.type, self.CODE.Parameter))
        self._add_basic_properties(entity_uri, entity)

        # Parameter modifiers
        self.graph.add((entity_uri, self.CODE.isRestParameter, Literal(entity.is_rest_parameter, datatype=XSD.boolean)))
        self.graph.add((entity_uri, self.CODE.isOptional, Literal(entity.type_info.is_optional if entity.type_info else False, datatype=XSD.boolean)))

        if entity.default_value:
            self.graph.add((entity_uri, self.CODE.hasDefaultValue,
                          Literal(entity.default_value, datatype=XSD.string)))

    def _add_import(self, entity_uri: URIRef, entity: ImportEntity):
        """Add import-specific properties"""
        self.graph.add((entity_uri, RDF.type, self.CODE.Import))

        # Import details
        self.graph.add((entity_uri, self.CODE.modulePath, Literal(entity.module_path, datatype=XSD.string)))
        self.graph.add((entity_uri, self.CODE.importType, Literal(entity.import_type, datatype=XSD.string)))
        self.graph.add((entity_uri, self.CODE.isTypeOnly, Literal(entity.is_type_only, datatype=XSD.boolean)))

        if entity.alias:
            self.graph.add((entity_uri, self.CODE.hasAlias, Literal(entity.alias, datatype=XSD.string)))

        # Imported symbols
        for symbol in entity.imported_symbols:
            self.graph.add((entity_uri, self.CODE.importsSymbol, Literal(symbol, datatype=XSD.string)))

    def _add_export(self, entity_uri: URIRef, entity: ExportEntity):
        """Add export-specific properties"""
        self.graph.add((entity_uri, RDF.type, self.CODE.Export))

        # Export details
        self.graph.add((entity_uri, self.CODE.exportType, Literal(entity.export_type, datatype=XSD.string)))
        self.graph.add((entity_uri, self.CODE.isReExport, Literal(entity.is_re_export, datatype=XSD.boolean)))

        if entity.alias:
            self.graph.add((entity_uri, self.CODE.hasAlias, Literal(entity.alias, datatype=XSD.string)))

        if entity.source_module:
            self.graph.add((entity_uri, self.CODE.fromModule, Literal(entity.source_module, datatype=XSD.string)))

    def _add_call_expression(self, entity_uri: URIRef, entity: CallExpressionEntity):
        """Add call expression properties"""
        self.graph.add((entity_uri, RDF.type, self.CODE.CallExpression))

        # Call details
        self.graph.add((entity_uri, self.CODE.callsFunction, Literal(entity.callee_name, datatype=XSD.string)))
        self.graph.add((entity_uri, self.CODE.isMethodCall, Literal(entity.is_method_call, datatype=XSD.boolean)))

        # Arguments
        for i, arg in enumerate(entity.arguments):
            arg_node = BNode()
            self.graph.add((entity_uri, self.CODE.hasArgument, arg_node))
            self.graph.add((arg_node, self.CODE.argumentPosition, Literal(i, datatype=XSD.integer)))
            self.graph.add((arg_node, self.CODE.argumentValue, Literal(arg, datatype=XSD.string)))

    def _add_relationships(self, entity: CodeEntity):
        """Add relationships between entities"""
        entity_uri = URIRef(entity.uri)

        # Function relationships
        if isinstance(entity, (FunctionEntity, MethodEntity)):
            # Function calls
            for call_uri in entity.calls:
                if call_uri in self._entity_map:
                    target_uri = URIRef(call_uri)
                    self.graph.add((entity_uri, self.CODE.calls, target_uri))
                    self.graph.add((target_uri, self.CODE.calledBy, entity_uri))

            # Parameters
            for param_uri in getattr(entity, 'parameter_uris', []):
                if param_uri in self._entity_map:
                    self.graph.add((entity_uri, self.CODE.hasParameter, URIRef(param_uri)))

        # Class relationships
        if isinstance(entity, ClassEntity):
            # Inheritance
            if entity.extends_class and entity.extends_class in self._entity_map:
                self.graph.add((entity_uri, self.CODE.extends, URIRef(entity.extends_class)))

            # Interface implementation
            for interface_uri in entity.implements_interfaces:
                if interface_uri in self._entity_map:
                    self.graph.add((entity_uri, self.CODE.implements, URIRef(interface_uri)))

            # Methods and properties
            for method_uri in entity.methods:
                if method_uri in self._entity_map:
                    self.graph.add((entity_uri, self.CODE.hasMethod, URIRef(method_uri)))
                    self.graph.add((URIRef(method_uri), self.CODE.memberOf, entity_uri))

            for prop_uri in entity.properties:
                if prop_uri in self._entity_map:
                    self.graph.add((entity_uri, self.CODE.hasProperty, URIRef(prop_uri)))
                    self.graph.add((URIRef(prop_uri), self.CODE.memberOf, entity_uri))

        # Module relationships
        if isinstance(entity, ModuleEntity):
            # Contains relationships
            for func_uri in entity.functions:
                if func_uri in self._entity_map:
                    self.graph.add((entity_uri, self.CODE.defines, URIRef(func_uri)))
                    self.graph.add((URIRef(func_uri), self.CODE.declaredIn, entity_uri))

            for class_uri in entity.classes:
                if class_uri in self._entity_map:
                    self.graph.add((entity_uri, self.CODE.defines, URIRef(class_uri)))
                    self.graph.add((URIRef(class_uri), self.CODE.declaredIn, entity_uri))

            # Export relationships
            for export_uri in entity.exports:
                if export_uri in self._entity_map:
                    self.graph.add((entity_uri, self.CODE.exports, URIRef(export_uri)))

            # Import relationships (module-to-module)
            for imported_module_uri in entity.imports:
                try:
                    self.graph.add((entity_uri, self.CODE.imports, URIRef(imported_module_uri)))
                except Exception:
                    pass

        # Call expression relationships
        if isinstance(entity, CallExpressionEntity):
            if entity.caller_uri and entity.caller_uri in self._entity_map:
                self.graph.add((entity_uri, self.CODE.madeBy, URIRef(entity.caller_uri)))

            if entity.callee_uri and entity.callee_uri in self._entity_map:
                self.graph.add((entity_uri, self.CODE.callsFunction, URIRef(entity.callee_uri)))
                self.graph.add((URIRef(entity.callee_uri), self.CODE.calledAt, entity_uri))

    def serialize(self, format: str = "turtle", destination: Optional[str] = None) -> str:
        """
        Serialize the graph to string or file
        Supported formats: turtle, xml, n3, nt, json-ld
        """
        if destination:
            self.graph.serialize(destination=destination, format=format)
            return f"Graph serialized to {destination}"
        else:
            return self.graph.serialize(format=format)

    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query against the graph"""
        results = []
        try:
            query_result = self.graph.query(sparql_query)
            for row in query_result:
                result_row = {}
                for var_name in query_result.vars:
                    result_row[str(var_name)] = str(row[var_name]) if row[var_name] else None
                results.append(result_row)
        except Exception as e:
            print(f"Query error: {e}")

        return results

    def get_entity_count(self) -> Dict[str, int]:
        """Get count of different entity types in the graph"""
        counts = {}

        # Count each type
        type_queries = {
            'modules': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a {self.CODE.Module} }}",
            'functions': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a {self.CODE.Function} }}",
            'classes': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a {self.CODE.Class} }}",
            'variables': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a {self.CODE.Variable} }}",
            'imports': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a {self.CODE.Import} }}",
            'exports': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a {self.CODE.Export} }}",
            'calls': f"SELECT (COUNT(?s) as ?count) WHERE {{ ?s a {self.CODE.CallExpression} }}"
        }

        for entity_type, query in type_queries.items():
            try:
                result = list(self.graph.query(query))
                counts[entity_type] = int(result[0][0]) if result else 0
            except:
                counts[entity_type] = 0

        return counts

    def validate_graph(self) -> List[str]:
        """Validate graph consistency and return list of issues"""
        issues = []

        # Check for orphaned entities (entities without proper type)
        orphan_query = """
        SELECT ?s WHERE {
            ?s code:hasName ?name .
            MINUS { ?s a ?type }
        }
        """

        try:
            orphans = list(self.graph.query(orphan_query, initBindings={'code': self.CODE}))
            if orphans:
                issues.append(f"Found {len(orphans)} entities without type")
        except Exception as e:
            issues.append(f"Validation query failed: {e}")

        # Check for missing required properties
        required_props_query = """
        SELECT ?s WHERE {
            ?s a code:CodeEntity .
            MINUS { ?s code:hasName ?name }
        }
        """

        try:
            missing_names = list(self.graph.query(required_props_query, initBindings={'code': self.CODE}))
            if missing_names:
                issues.append(f"Found {len(missing_names)} entities without names")
        except Exception as e:
            issues.append(f"Required property validation failed: {e}")

        return issues


def create_ontology_builder(ontology_path: Optional[str] = None) -> OntologyBuilder:
    """Factory function to create ontology builder"""
    return OntologyBuilder(ontology_path)
