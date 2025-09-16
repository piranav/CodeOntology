"""
Language-agnostic AST parser using tree-sitter.
Focuses on JavaScript/TypeScript with extensibility for other languages.
"""

import tree_sitter
from tree_sitter import Parser
try:
    from tree_sitter_languages import get_language  # optional
except Exception:
    get_language = None
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import hashlib

from ..models.code_entities import (
    CodeEntity, FunctionEntity, ClassEntity, VariableEntity,
    ImportEntity, ExportEntity, ModuleEntity, SourceLocation,
    ParameterEntity, MethodEntity, PropertyEntity, InterfaceEntity,
    TypeInfo, CallExpressionEntity, create_entity
)


class ASTNode:
    """Wrapper for tree-sitter nodes with additional metadata"""

    def __init__(self, node: tree_sitter.Node, source_code: str):
        self.node = node
        self.source_code = source_code
        self._text = None

    @property
    def text(self) -> str:
        """Get text content of this node"""
        if self._text is None:
            self._text = self.source_code[self.node.start_byte:self.node.end_byte]
        return self._text

    @property
    def type(self) -> str:
        return self.node.type

    @property
    def children(self) -> List['ASTNode']:
        return [ASTNode(child, self.source_code) for child in self.node.children]

    def find_child(self, node_type: str) -> Optional['ASTNode']:
        """Find first child of given type"""
        for child in self.children:
            if child.type == node_type:
                return child
        return None

    def find_children(self, node_type: str) -> List['ASTNode']:
        """Find all children of given type"""
        return [child for child in self.children if child.type == node_type]

    def get_location(self) -> SourceLocation:
        """Convert tree-sitter position to SourceLocation"""
        return SourceLocation(
            file_path="",  # Will be set by parser
            line_number=self.node.start_point[0] + 1,
            column=self.node.start_point[1],
            end_line=self.node.end_point[0] + 1,
            end_column=self.node.end_point[1]
        )


class ASTParser:
    """
    AST parser supporting JavaScript and TypeScript
    Extracts code entities while preserving structural relationships
    """

    def __init__(self):
        self.use_fallback = False
        if get_language is None:
            self.use_fallback = True
        else:
            try:
                self.js_language = get_language('javascript')
                # Prefer 'tsx' for .tsx, 'typescript' otherwise
                self.ts_language = get_language('typescript')

                # Language detection by file extension
                self.language_map = {
                    '.js': self.js_language,
                    '.jsx': self.js_language,
                    '.ts': self.ts_language,
                    '.tsx': self.ts_language,
                }

                # Parser cache for reuse
                self.parsers = {}
            except Exception:
                # Fallback to regex-based extraction when tree-sitter languages are unavailable
                self.use_fallback = True

    def _get_parser(self, language) -> Parser:
        """Get or create parser for language"""
        lang_id = str(language)
        if lang_id not in self.parsers:
            parser = Parser()
            parser.set_language(language)
            self.parsers[lang_id] = parser
        return self.parsers[lang_id]

    def parse_file(self, filepath: str) -> Tuple[ASTNode, List[CodeEntity]]:
        """
        Parse a source file and extract entities
        Returns the AST root and list of extracted entities
        """
        file_path = Path(filepath)

        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        if self.use_fallback:
            entities = self._regex_extract_entities(source_code, str(file_path))
            return None, entities

        # Detect language
        language = self.language_map.get(file_path.suffix, self.js_language)
        parser = self._get_parser(language)

        # Parse to AST
        tree = parser.parse(source_code.encode('utf-8'))
        root_node = ASTNode(tree.root_node, source_code)

        # Extract entities
        entities = self.extract_entities(root_node, str(file_path))

        return root_node, entities

    # -----------------------------
    # Fallback regex-based extractor
    # -----------------------------
    def _regex_extract_entities(self, source: str, file_path: str) -> List[CodeEntity]:
        import re

        entities: List[CodeEntity] = []

        # Module entity
        module_entity = ModuleEntity(
            name=Path(file_path).stem,
            location=SourceLocation(file_path=file_path, line_number=1, column=0),
            body_hash=self._generate_content_hash(source)
        )
        entities.append(module_entity)

        # Patterns
        func_decl_re = re.compile(r"^\s*function\s+([A-Za-z_][\w]*)\s*\(([^)]*)\)", re.MULTILINE)
        class_decl_re = re.compile(r"^\s*class\s+([A-Za-z_][\w]*)\s*\{", re.MULTILINE)
        method_re = re.compile(r"^\s*([A-Za-z_#][\w]*)\s*\(([^)]*)\)\s*\{", re.MULTILINE)
        import_re = re.compile(r"^\s*import\s+[^;]*?from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
        export_func_re = re.compile(r"^\s*export\s+function\s+([A-Za-z_][\w]*)", re.MULTILINE)
        export_default_func_re = re.compile(r"^\s*export\s+default\s+function\s+([A-Za-z_][\w]*)?", re.MULTILINE)

        lines = source.splitlines()

        # Functions
        for m in func_decl_re.finditer(source):
            name = m.group(1)
            line_number = source.count('\n', 0, m.start()) + 1
            loc = SourceLocation(file_path=file_path, line_number=line_number, column=0)
            params_text = m.group(2).strip()
            params = []
            if params_text:
                for p in [p.strip() for p in params_text.split(',') if p.strip()]:
                    # sanitize param name (strip destructuring braces and defaults)
                    pname = p.split('=')[0].strip()
                    pname = pname.strip('{}[]() ')
                    if not pname:
                        continue
                    params.append(ParameterEntity(name=pname, location=loc))

            func = FunctionEntity(
                name=name,
                location=loc,
                parameters=params,
                scope='global',
                body_hash=self._generate_content_hash(m.group(0))
            )
            entities.extend(params)
            entities.append(func)
            module_entity.functions.append(func.uri)

            # Extract calls in function body (naive brace matching)
            body_start = source.find('{', m.end())
            if body_start != -1:
                brace = 1
                i = body_start + 1
                while i < len(source) and brace > 0:
                    ch = source[i]
                    if ch == '{':
                        brace += 1
                    elif ch == '}':
                        brace -= 1
                    i += 1
                body = source[body_start:i-1] if brace == 0 else source[body_start:]
                call_re = re.compile(r"\b([A-Za-z_][\w]*)\s*\(")
                call_names = []
                for cm_ in call_re.finditer(body):
                    callee = cm_.group(1)
                    if callee not in {'if', 'for', 'while', 'switch', 'catch', 'function', 'class', 'return', 'console'}:
                        call_names.append(callee)
                setattr(func, '_call_names', call_names)

        # Exports
        exported = set()
        for m in export_func_re.finditer(source):
            exported.add(m.group(1))
        for m in export_default_func_re.finditer(source):
            if m.group(1):
                exported.add(m.group(1))

        # Mark exported functions
        by_name = {e.name: e for e in entities if isinstance(e, FunctionEntity)}
        for name in exported:
            if name in by_name:
                by_name[name].is_exported = True

        # Imports
        for m in import_re.finditer(source):
            module_spec = m.group(1)
            line_number = source.count('\n', 0, m.start()) + 1
            imp = ImportEntity(
                name=f"import_{Path(module_spec).stem}",
                location=SourceLocation(file_path=file_path, line_number=line_number, column=0),
                module_path=module_spec,
            )
            entities.append(imp)

        # Classes and methods (naive - scans class blocks)
        for cm in class_decl_re.finditer(source):
            class_name = cm.group(1)
            class_start = cm.end()
            line_number = source.count('\n', 0, cm.start()) + 1
            loc = SourceLocation(file_path=file_path, line_number=line_number, column=0)
            cls = ClassEntity(name=class_name, location=loc, body_hash=self._generate_content_hash(cm.group(0)))
            entities.append(cls)
            module_entity.classes.append(cls.uri)

            # Find methods within a simple scope until matching closing brace count
            brace = 1
            i = class_start
            while i < len(source) and brace > 0:
                ch = source[i]
                if ch == '{':
                    brace += 1
                elif ch == '}':
                    brace -= 1
                i += 1
            class_block = source[class_start:i-1] if brace == 0 else source[class_start:]
            for mm in method_re.finditer(class_block):
                mname = mm.group(1)
                mline = source.count('\n', 0, class_start + mm.start()) + 1
                mloc = SourceLocation(file_path=file_path, line_number=mline, column=0)
                params_text = mm.group(2).strip()
                params = []
                if params_text:
                    for p in [p.strip() for p in params_text.split(',') if p.strip()]:
                        pname = p.split('=')[0].strip()
                        pname = pname.strip('{}[]() ')
                        if not pname:
                            continue
                        params.append(ParameterEntity(name=pname, location=mloc))
                meth = MethodEntity(name=mname, location=mloc, parameters=params, parent_class_uri=cls.uri)
                entities.extend(params)
                entities.append(meth)
                cls.methods.append(meth.uri)

                # Method body
                body_start = class_block.find('{', mm.end())
                if body_start != -1:
                    brace = 1
                    i2 = body_start + 1
                    while i2 < len(class_block) and brace > 0:
                        ch2 = class_block[i2]
                        if ch2 == '{':
                            brace += 1
                        elif ch2 == '}':
                            brace -= 1
                        i2 += 1
                    body = class_block[body_start:i2-1] if brace == 0 else class_block[body_start:]
                    call_re = re.compile(r"\b([A-Za-z_][\w]*)\s*\(")
                    call_names = []
                    for cm_ in call_re.finditer(body):
                        callee = cm_.group(1)
                        if callee not in {'if', 'for', 'while', 'switch', 'catch', 'function', 'class', 'return', 'console'}:
                            call_names.append(callee)
                    setattr(meth, '_call_names', call_names)

        # Resolve call relationships by name
        by_name_all = {e.name: e for e in entities if isinstance(e, (FunctionEntity, MethodEntity))}
        for e in [x for x in entities if isinstance(x, (FunctionEntity, MethodEntity))]:
            names = getattr(e, '_call_names', [])
            for callee_name in names:
                callee = by_name_all.get(callee_name)
                if callee:
                    e.calls.append(callee.uri)
                    callee.called_by.append(e.uri)

        return entities

    def extract_entities(self, root_node: ASTNode, file_path: str) -> List[CodeEntity]:
        """
        Extract all code entities from AST
        Preserves parent-child relationships and cross-references
        """
        entities = []

        # Create module entity first
        module_entity = self._create_module_entity(root_node, file_path)
        entities.append(module_entity)

        # Track context for scoping
        context = {
            'module': module_entity,
            'current_class': None,
            'current_function': None,
            'file_path': file_path
        }

        # Recursively extract entities
        self._extract_from_node(root_node, entities, context)

        # Post-process to establish relationships
        self._establish_relationships(entities)

        return entities

    def _create_module_entity(self, root_node: ASTNode, file_path: str) -> ModuleEntity:
        """Create entity for the module/file"""
        location = SourceLocation(
            file_path=file_path,
            line_number=1,
            column=0
        )

        return ModuleEntity(
            name=Path(file_path).stem,
            location=location,
            body_hash=self._generate_content_hash(root_node.text)
        )

    def _extract_from_node(self, node: ASTNode, entities: List[CodeEntity], context: Dict[str, Any]):
        """Recursively extract entities from AST node"""

        # Set file path in location for all nodes
        if hasattr(node, 'get_location'):
            location = node.get_location()
            location.file_path = context['file_path']

        # Handle different node types
        if node.type == 'function_declaration':
            entity = self._extract_function(node, context)
            entities.append(entity)
            # Add parameter nodes as entities
            for p in getattr(entity, 'parameters', []):
                entities.append(p)
            # Attach to module
            context['module'].functions.append(entity.uri)

            # Update context for nested extraction
            old_function = context['current_function']
            context['current_function'] = entity

            # Extract function body
            body = node.find_child('statement_block')
            if body:
                self._extract_from_node(body, entities, context)

            context['current_function'] = old_function

        elif node.type == 'arrow_function':
            entity = self._extract_arrow_function(node, context)
            entities.append(entity)
            for p in getattr(entity, 'parameters', []):
                entities.append(p)
            context['module'].functions.append(entity.uri)

        elif node.type == 'class_declaration':
            entity = self._extract_class(node, context)
            entities.append(entity)
            context['module'].classes.append(entity.uri)

            # Update context for class members
            old_class = context['current_class']
            context['current_class'] = entity

            # Extract class body
            body = node.find_child('class_body')
            if body:
                self._extract_from_node(body, entities, context)

            context['current_class'] = old_class

        elif node.type == 'method_definition':
            entity = self._extract_method(node, context)
            entities.append(entity)

        elif node.type == 'variable_declaration':
            var_entities = self._extract_variables(node, context)
            entities.extend(var_entities)
            for v in var_entities:
                context['module'].variables.append(v.uri)

        elif node.type == 'import_statement':
            entity = self._extract_import(node, context)
            entities.append(entity)

        elif node.type == 'export_statement':
            entity = self._extract_export(node, context)
            entities.append(entity)
            context['module'].exports.append(entity.uri)

        elif node.type in ['interface_declaration', 'type_alias_declaration']:
            entity = self._extract_interface(node, context)
            entities.append(entity)
            context['module'].interfaces.append(entity.uri)

        elif node.type == 'call_expression':
            entity = self._extract_call_expression(node, context)
            entities.append(entity)

        # Recursively process children
        for child in node.children:
            self._extract_from_node(child, entities, context)

    def _extract_function(self, node: ASTNode, context: Dict[str, Any]) -> FunctionEntity:
        """Extract function declaration"""
        name_node = node.find_child('identifier')
        name = name_node.text if name_node else '<anonymous>'

        location = node.get_location()
        location.file_path = context['file_path']

        # Extract parameters
        parameters = self._extract_parameters(node)
        # Ensure parameter locations include file path
        for p in parameters:
            if not p.location.file_path:
                p.location.file_path = context['file_path']

        # Extract function modifiers
        is_async = self._has_modifier(node, 'async')
        is_generator = self._has_modifier(node, 'generator')

        # Determine if exported
        is_exported, is_default_export = self._check_export_status(node, context)

        func = FunctionEntity(
            name=name,
            location=location,
            parameters=parameters,
            is_async=is_async,
            is_generator=is_generator,
            is_exported=is_exported,
            is_default_export=is_default_export,
            body_hash=self._generate_content_hash(node.text),
            scope=self._determine_scope(context)
        )

        # Also add ParameterEntity nodes to global list by stashing their URIs on the function
        try:
            func.parameter_uris = [param.uri for param in parameters]
        except Exception:
            pass

        return func

    def _extract_arrow_function(self, node: ASTNode, context: Dict[str, Any]) -> FunctionEntity:
        """Extract arrow function expression"""
        location = node.get_location()
        location.file_path = context['file_path']

        parameters = self._extract_parameters(node)
        for p in parameters:
            if not p.location.file_path:
                p.location.file_path = context['file_path']
        is_async = self._has_modifier(node, 'async')

        func = FunctionEntity(
            name='<arrow_function>',
            location=location,
            parameters=parameters,
            is_async=is_async,
            is_arrow_function=True,
            body_hash=self._generate_content_hash(node.text),
            scope=self._determine_scope(context)
        )
        try:
            func.parameter_uris = [param.uri for param in parameters]
        except Exception:
            pass
        return func

    def _extract_class(self, node: ASTNode, context: Dict[str, Any]) -> ClassEntity:
        """Extract class declaration"""
        name_node = node.find_child('identifier')
        name = name_node.text if name_node else '<anonymous>'

        location = node.get_location()
        location.file_path = context['file_path']

        # Extract inheritance
        extends_class = None
        heritage_clause = node.find_child('class_heritage')
        if heritage_clause:
            extends_node = heritage_clause.find_child('identifier')
            extends_class = extends_node.text if extends_node else None

        # Check export status
        is_exported, is_default_export = self._check_export_status(node, context)

        return ClassEntity(
            name=name,
            location=location,
            extends_class=extends_class,
            is_exported=is_exported,
            is_default_export=is_default_export,
            body_hash=self._generate_content_hash(node.text)
        )

    def _extract_method(self, node: ASTNode, context: Dict[str, Any]) -> MethodEntity:
        """Extract class method"""
        name_node = node.find_child('property_identifier') or node.find_child('identifier')
        name = name_node.text if name_node else '<anonymous>'

        location = node.get_location()
        location.file_path = context['file_path']

        parameters = self._extract_parameters(node)

        # Method modifiers
        is_static = self._has_modifier(node, 'static')
        is_private = name.startswith('#') or self._has_modifier(node, 'private')
        is_async = self._has_modifier(node, 'async')
        is_generator = self._has_modifier(node, 'generator')
        is_constructor = name == 'constructor'

        parent_class_uri = None
        if context['current_class']:
            parent_class_uri = context['current_class'].uri

        return MethodEntity(
            name=name,
            location=location,
            parameters=parameters,
            is_static=is_static,
            is_private=is_private,
            is_async=is_async,
            is_generator=is_generator,
            is_constructor=is_constructor,
            parent_class_uri=parent_class_uri,
            body_hash=self._generate_content_hash(node.text)
        )

    def _extract_variables(self, node: ASTNode, context: Dict[str, Any]) -> List[VariableEntity]:
        """Extract variable declarations"""
        variables = []

        # Get declaration kind (const, let, var)
        kind = node.find_child('const') or node.find_child('let') or node.find_child('var')
        kind_text = kind.text if kind else 'var'

        # Find all variable declarators
        declarators = node.find_children('variable_declarator')

        for declarator in declarators:
            name_node = declarator.find_child('identifier')
            if not name_node:
                continue

            name = name_node.text
            location = declarator.get_location()
            location.file_path = context['file_path']

            # Get initialization value if present
            init_value = None
            if len(declarator.children) > 1:  # Has initializer
                init_value = declarator.children[-1].text

            variables.append(VariableEntity(
                name=name,
                location=location,
                is_const=kind_text == 'const',
                is_let=kind_text == 'let',
                is_var=kind_text == 'var',
                initialization_value=init_value,
                scope=self._determine_scope(context)
            ))

        return variables

    def _extract_import(self, node: ASTNode, context: Dict[str, Any]) -> ImportEntity:
        """Extract import statement"""
        location = node.get_location()
        location.file_path = context['file_path']

        # Get source module
        source_node = node.find_child('string')
        module_spec = source_node.text.strip("\"'{};") if source_node else ""

        # Resolve to an absolute path if relative
        resolved_uri = None
        try:
            base = Path(context['file_path']).parent
            if module_spec.startswith('.'):
                abs_path = (base / module_spec).resolve()
                # Try to ensure an extension
                if abs_path.is_dir():
                    for idx in ['index.ts', 'index.tsx', 'index.js', 'index.jsx']:
                        candidate = abs_path / idx
                        if candidate.exists():
                            abs_path = candidate
                            break
                else:
                    if not abs_path.suffix:
                        for ext in ['.ts', '.tsx', '.js', '.jsx']:
                            p = Path(str(abs_path) + ext)
                            if p.exists():
                                abs_path = p
                                break
                resolved_uri = f"http://codebase.local/{abs_path}#module"
        except Exception:
            resolved_uri = None

        # Extract imported symbols
        imported_symbols = []
        import_clause = node.find_child('import_clause')
        if import_clause:
            # Named imports
            named_imports = import_clause.find_child('named_imports')
            if named_imports:
                for spec in named_imports.find_children('import_specifier'):
                    symbol = spec.find_child('identifier')
                    if symbol:
                        imported_symbols.append(symbol.text)

            # Default import
            default_import = import_clause.find_child('identifier')
            if default_import:
                imported_symbols.append(default_import.text)

        imp = ImportEntity(
            name=f"import_{Path(module_spec).stem}",
            location=location,
            module_path=module_spec,
            imported_symbols=imported_symbols
        )

        # Record on current module
        if resolved_uri:
            try:
                context['module'].imports.append(resolved_uri)
            except Exception:
                pass

        return imp

    def _extract_export(self, node: ASTNode, context: Dict[str, Any]) -> ExportEntity:
        """Extract export statement"""
        location = node.get_location()
        location.file_path = context['file_path']

        # Determine export type and target
        if node.find_child('default'):
            export_type = 'default'
            # Find what's being exported
            exported_node = node.children[-1]  # Usually the last child
            symbol_name = self._get_exported_symbol_name(exported_node)
        else:
            export_type = 'named'
            symbol_name = self._get_exported_symbol_name(node)

        return ExportEntity(
            name=f"export_{symbol_name}",
            location=location,
            exported_symbol_uri="",  # Will be resolved later
            export_type=export_type
        )

    def _extract_interface(self, node: ASTNode, context: Dict[str, Any]) -> InterfaceEntity:
        """Extract TypeScript interface"""
        name_node = node.find_child('type_identifier') or node.find_child('identifier')
        name = name_node.text if name_node else '<anonymous>'

        location = node.get_location()
        location.file_path = context['file_path']

        is_exported, _ = self._check_export_status(node, context)

        return InterfaceEntity(
            name=name,
            location=location,
            is_exported=is_exported,
            body_hash=self._generate_content_hash(node.text)
        )

    def _extract_call_expression(self, node: ASTNode, context: Dict[str, Any]) -> CallExpressionEntity:
        """Extract function call expression"""
        location = node.get_location()
        location.file_path = context['file_path']

        # Get function being called
        function_node = node.children[0] if node.children else None
        callee_name = function_node.text if function_node else '<unknown>'

        # Get caller context
        caller_uri = ""
        if context['current_function']:
            caller_uri = context['current_function'].uri
        elif context['current_class']:
            caller_uri = context['current_class'].uri

        # Extract arguments
        arguments = []
        args_node = node.find_child('arguments')
        if args_node:
            for arg in args_node.children:
                if arg.type != '(' and arg.type != ')' and arg.type != ',':
                    arguments.append(arg.text)

        return CallExpressionEntity(
            name=f"call_{callee_name}",
            location=location,
            caller_uri=caller_uri,
            callee_name=callee_name,
            arguments=arguments,
            is_method_call='.' in callee_name or '->' in callee_name
        )

    def _extract_parameters(self, node: ASTNode) -> List[ParameterEntity]:
        """Extract function parameters"""
        parameters = []

        params_node = node.find_child('formal_parameters')
        if not params_node:
            return parameters

        for param_node in params_node.children:
            if param_node.type == 'identifier':
                loc = param_node.get_location()
                param = ParameterEntity(
                    name=param_node.text,
                    location=loc
                )
                parameters.append(param)
            elif param_node.type == 'rest_parameter':
                # Handle ...args
                identifier = param_node.find_child('identifier')
                if identifier:
                    loc = param_node.get_location()
                    param = ParameterEntity(
                        name=identifier.text,
                        location=loc,
                        is_rest_parameter=True
                    )
                    parameters.append(param)

        return parameters

    def _has_modifier(self, node: ASTNode, modifier: str) -> bool:
        """Check if node has specific modifier"""
        for child in node.children:
            if child.text == modifier:
                return True
        return False

    def _check_export_status(self, node: ASTNode, context: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if node is exported and if it's default export"""
        # This would need to be enhanced to check parent nodes for export statements
        return False, False

    def _determine_scope(self, context: Dict[str, Any]) -> str:
        """Determine the scope context"""
        if context['current_function']:
            return 'function'
        elif context['current_class']:
            return 'class'
        else:
            return 'global'

    def _get_exported_symbol_name(self, node: ASTNode) -> str:
        """Get name of exported symbol"""
        if hasattr(node, 'text'):
            return node.text[:50]  # Truncate for readability
        return '<unknown>'

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content change detection"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _establish_relationships(self, entities: List[CodeEntity]):
        """Post-process entities to establish cross-references"""
        # Create lookup maps
        entities_by_name = {}
        entities_by_uri = {}

        for entity in entities:
            entities_by_name[entity.name] = entity
            if entity.uri:
                entities_by_uri[entity.uri] = entity

        # Establish function call relationships
        for entity in entities:
            if isinstance(entity, CallExpressionEntity):
                # Try to resolve callee
                callee = entities_by_name.get(entity.callee_name)
                if callee and isinstance(callee, (FunctionEntity, MethodEntity)):
                    entity.callee_uri = callee.uri
                    callee.called_by.append(entity.caller_uri)

                    # Add to caller's calls list
                    caller = entities_by_uri.get(entity.caller_uri)
                    if caller and isinstance(caller, (FunctionEntity, MethodEntity)):
                        caller.calls.append(callee.uri)

        # Establish class-member relationships
        for entity in entities:
            if isinstance(entity, (MethodEntity, PropertyEntity)) and entity.parent_class_uri:
                parent_class = entities_by_uri.get(entity.parent_class_uri)
                if isinstance(parent_class, ClassEntity):
                    if isinstance(entity, MethodEntity):
                        parent_class.methods.append(entity.uri)
                    elif isinstance(entity, PropertyEntity):
                        parent_class.properties.append(entity.uri)


def create_ast_parser() -> ASTParser:
    """Factory function to create AST parser"""
    return ASTParser()
