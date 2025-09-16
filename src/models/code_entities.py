"""
Code entity models using Pydantic for validation and type safety.
Maps AST nodes to ontology entities with unique URI generation.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import uuid
import re


class SourceLocation(BaseModel):
    """Represents location in source code"""
    file_path: str
    line_number: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    @property
    def range_string(self) -> str:
        if self.end_line and self.end_column:
            return f"{self.line_number}:{self.column}-{self.end_line}:{self.end_column}"
        return f"{self.line_number}:{self.column}"


class CodeEntity(BaseModel):
    """Base class for all code entities with common properties"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    location: SourceLocation
    uri: Optional[str] = None
    docstring: Optional[str] = None
    comments: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    body_hash: Optional[str] = None  # Hash of the entity's body for change detection

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:  # pydantic v2 hook
        if not self.uri:
            self.uri = self._generate_uri()

    def _generate_uri(self) -> str:
        """Generate unique URI for this entity"""
        base_uri = f"http://codebase.local/{self.location.file_path}"
        entity_type = self.__class__.__name__.replace("Entity", "").lower()
        # Sanitize name to be URI-friendly
        raw = self.name or 'unknown'
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", raw)
        if not safe_name:
            safe_name = 'unknown'
        return f"{base_uri}#{entity_type}_{safe_name}_{self.location.line_number}"

    def generate_body_hash(self, body_content: str) -> str:
        """Generate hash for change detection"""
        return hashlib.sha256(body_content.encode()).hexdigest()[:16]


class TypeInfo(BaseModel):
    """Type information for variables, parameters, returns"""
    name: str
    is_optional: bool = False
    is_array: bool = False
    is_generic: bool = False
    generic_params: List[str] = Field(default_factory=list)


class ParameterEntity(CodeEntity):
    """Function or method parameter"""
    type_info: Optional[TypeInfo] = None
    default_value: Optional[str] = None
    is_rest_parameter: bool = False  # ...args
    is_destructured: bool = False


class FunctionEntity(CodeEntity):
    """Function declaration or expression"""
    parameters: List[ParameterEntity] = Field(default_factory=list)
    return_type: Optional[TypeInfo] = None
    is_async: bool = False
    is_generator: bool = False
    is_arrow_function: bool = False
    is_exported: bool = False
    is_default_export: bool = False
    scope: str = "global"  # global, class, function

    # Relationships (URIs of related entities)
    calls: List[str] = Field(default_factory=list)  # Functions this calls
    called_by: List[str] = Field(default_factory=list)  # Functions that call this
    accesses_variables: List[str] = Field(default_factory=list)  # Variables accessed


class MethodEntity(FunctionEntity):
    """Class method"""
    is_static: bool = False
    is_private: bool = False
    is_protected: bool = False
    is_constructor: bool = False
    is_getter: bool = False
    is_setter: bool = False
    parent_class_uri: Optional[str] = None


class VariableEntity(CodeEntity):
    """Variable declaration"""
    type_info: Optional[TypeInfo] = None
    is_const: bool = False
    is_let: bool = False
    is_var: bool = False
    initialization_value: Optional[str] = None
    scope: str = "global"  # global, function, block, class

    # Relationships
    used_by: List[str] = Field(default_factory=list)  # Functions that use this variable


class PropertyEntity(VariableEntity):
    """Class property"""
    is_static: bool = False
    is_private: bool = False
    is_protected: bool = False
    is_readonly: bool = False
    parent_class_uri: Optional[str] = None


class ClassEntity(CodeEntity):
    """Class declaration"""
    methods: List[str] = Field(default_factory=list)  # URIs of method entities
    properties: List[str] = Field(default_factory=list)  # URIs of property entities
    constructor_uri: Optional[str] = None
    extends_class: Optional[str] = None  # URI of parent class
    implements_interfaces: List[str] = Field(default_factory=list)  # URIs of interfaces
    is_abstract: bool = False
    is_exported: bool = False
    is_default_export: bool = False

    # Type parameters for generics
    type_parameters: List[str] = Field(default_factory=list)


class InterfaceEntity(CodeEntity):
    """TypeScript interface"""
    methods: List[str] = Field(default_factory=list)
    properties: List[str] = Field(default_factory=list)
    extends_interfaces: List[str] = Field(default_factory=list)
    is_exported: bool = False
    type_parameters: List[str] = Field(default_factory=list)


class ModuleEntity(CodeEntity):
    """Module/file entity"""
    imports: List[str] = Field(default_factory=list)  # URIs of imported entities
    exports: List[str] = Field(default_factory=list)  # URIs of exported entities
    functions: List[str] = Field(default_factory=list)  # URIs of functions
    classes: List[str] = Field(default_factory=list)  # URIs of classes
    interfaces: List[str] = Field(default_factory=list)  # URIs of interfaces
    variables: List[str] = Field(default_factory=list)  # URIs of variables

    # Module system info
    module_type: str = "commonjs"  # commonjs, es6, umd
    dependencies: List[str] = Field(default_factory=list)  # External dependencies

    def _generate_uri(self) -> str:
        return f"http://codebase.local/{self.location.file_path}#module"


class ImportEntity(CodeEntity):
    """Import statement"""
    module_path: str
    imported_symbols: List[str] = Field(default_factory=list)
    import_type: str = "named"  # named, default, namespace, dynamic
    alias: Optional[str] = None
    is_type_only: bool = False  # TypeScript type-only imports


class ExportEntity(CodeEntity):
    """Export statement"""
    exported_symbol_uri: str
    export_type: str = "named"  # named, default, namespace
    alias: Optional[str] = None
    is_re_export: bool = False
    source_module: Optional[str] = None


class CallExpressionEntity(CodeEntity):
    """Function call expression"""
    caller_uri: str  # URI of function that makes this call
    callee_uri: Optional[str] = None  # URI of function being called (if resolvable)
    callee_name: str  # Name of function being called
    arguments: List[str] = Field(default_factory=list)  # Argument expressions
    is_method_call: bool = False
    receiver_type: Optional[str] = None  # For method calls, type of receiver


# Type aliases for convenience
EntityType = Union[
    FunctionEntity, MethodEntity, ClassEntity, InterfaceEntity,
    VariableEntity, PropertyEntity, ModuleEntity, ImportEntity,
    ExportEntity, ParameterEntity, CallExpressionEntity
]

# Entity type registry for factory pattern
ENTITY_TYPES = {
    'function': FunctionEntity,
    'method': MethodEntity,
    'class': ClassEntity,
    'interface': InterfaceEntity,
    'variable': VariableEntity,
    'property': PropertyEntity,
    'module': ModuleEntity,
    'import': ImportEntity,
    'export': ExportEntity,
    'parameter': ParameterEntity,
    'call': CallExpressionEntity,
}


def create_entity(entity_type: str, **kwargs) -> EntityType:
    """Factory function to create entities"""
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unknown entity type: {entity_type}")
    return ENTITY_TYPES[entity_type](**kwargs)
