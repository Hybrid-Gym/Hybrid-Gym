"""Test script to verify the fixes for attribute calls and decorator dependencies."""

import ast
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_dataset import CallLocationExtractor, RepoAnalyzer


def test_attribute_call_column_offset():
    """Test that attribute calls have correct column offset pointing to the attribute name."""
    code = '''
def my_function():
    self.helper()
    obj.method()
    module.submodule.func()
'''
    tree = ast.parse(code)
    func_node = tree.body[0]

    extractor = CallLocationExtractor(func_node.lineno)
    for child in func_node.body:
        extractor.visit(child)

    print("=== Test 1: Attribute Call Column Offset ===")
    print(f"Source code lines:")
    for i, line in enumerate(code.split('\n'), 1):
        print(f"  {i}: {line}")
    print()

    for call in extractor.calls:
        name = call['name']
        line = call['line']
        col = call['column']

        # Get the actual source line
        source_lines = code.split('\n')
        source_line = source_lines[line - 1] if line <= len(source_lines) else ""

        # Check if column points to the attribute name
        if col < len(source_line):
            char_at_col = source_line[col:col+len(name)]
        else:
            char_at_col = "<out of bounds>"

        matches = char_at_col == name
        status = "PASS" if matches else "FAIL"

        print(f"  [{status}] Call to '{name}' at line {line}, col {col}")
        print(f"         Source line: '{source_line}'")
        print(f"         Char at col {col}: '{char_at_col}' (expected '{name}')")
        print()

    return all(
        code.split('\n')[call['line'] - 1][call['column']:call['column']+len(call['name'])] == call['name']
        for call in extractor.calls
    )


def test_decorator_dependencies():
    """Test that calls inside decorators are captured."""
    code = '''
def get_config():
    pass

def register(x):
    pass

@register(get_config())
def my_function():
    pass
'''
    tree = ast.parse(code)
    # Find my_function (the decorated one)
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'my_function':
            func_node = node
            break

    print("=== Test 2: Decorator Dependencies ===")
    print(f"Source code:")
    for i, line in enumerate(code.strip().split('\n'), 1):
        print(f"  {i}: {line}")
    print()

    extractor = CallLocationExtractor(func_node.lineno)

    # Visit decorators
    for decorator in func_node.decorator_list:
        extractor.visit(decorator)

    # Visit body
    for child in func_node.body:
        extractor.visit(child)

    print(f"  Extracted calls from my_function (including decorators):")
    for call in extractor.calls:
        print(f"    - {call['name']} at line {call['line']}, col {call['column']}")

    # Check if get_config was captured
    call_names = [c['name'] for c in extractor.calls]
    has_register = 'register' in call_names
    has_get_config = 'get_config' in call_names

    print()
    print(f"  [{'PASS' if has_register else 'FAIL'}] Decorator 'register' captured: {has_register}")
    print(f"  [{'PASS' if has_get_config else 'FAIL'}] Decorator arg 'get_config' captured: {has_get_config}")
    print()

    return has_register and has_get_config


def test_jedi_resolution():
    """Test end-to-end Jedi resolution with a real temp repo."""
    print("=== Test 3: Jedi Resolution (End-to-End) ===")

    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a helper module
        helper_code = '''
def helper_function():
    """A helper function."""
    return 42

class HelperClass:
    def method(self):
        return "hello"
'''
        helper_path = os.path.join(tmpdir, 'helper.py')
        with open(helper_path, 'w') as f:
            f.write(helper_code)

        # Create a main module that uses the helper
        main_code = '''
from helper import helper_function, HelperClass

def my_function():
    result = helper_function()
    obj = HelperClass()
    return result
'''
        main_path = os.path.join(tmpdir, 'main.py')
        with open(main_path, 'w') as f:
            f.write(main_code)

        print(f"  Created temp repo at: {tmpdir}")
        print(f"  helper.py:")
        for i, line in enumerate(helper_code.strip().split('\n'), 1):
            print(f"    {i}: {line}")
        print(f"  main.py:")
        for i, line in enumerate(main_code.strip().split('\n'), 1):
            print(f"    {i}: {line}")
        print()

        # Analyze
        analyzer = RepoAnalyzer(tmpdir)
        analyzer.collect_all_definitions()

        print(f"  Collected definitions:")
        for (rel_path, line), def_info in analyzer.definitions_by_location.items():
            print(f"    - {def_info['name']} ({def_info['type']}) in {rel_path}:{line}")
        print()

        # Parse main.py and find my_function
        with open(main_path) as f:
            tree = ast.parse(f.read())

        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'my_function':
                func_node = node
                break

        # Get dependencies
        deps = analyzer.get_function_dependencies(func_node, 'main.py')

        print(f"  Dependencies of my_function:")
        for dep in deps:
            print(f"    - {dep['name']} ({dep['type']}) in {dep['file_path']}:{dep['line_start']}")
        print()

        dep_names = [d['name'] for d in deps]
        has_helper_func = 'helper_function' in dep_names
        has_helper_class = 'HelperClass' in dep_names

        print(f"  [{'PASS' if has_helper_func else 'FAIL'}] Resolved helper_function: {has_helper_func}")
        print(f"  [{'PASS' if has_helper_class else 'FAIL'}] Resolved HelperClass: {has_helper_class}")
        print()

        return has_helper_func and has_helper_class


def test_method_call_resolution():
    """Test that method calls like self.method() are resolved."""
    print("=== Test 4: Method Call Resolution (self.method) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        code = '''
class MyClass:
    def helper(self):
        return 42

    def main_method(self):
        result = self.helper()
        return result
'''
        file_path = os.path.join(tmpdir, 'myclass.py')
        with open(file_path, 'w') as f:
            f.write(code)

        print(f"  Source code:")
        for i, line in enumerate(code.strip().split('\n'), 1):
            print(f"    {i}: {line}")
        print()

        analyzer = RepoAnalyzer(tmpdir)
        analyzer.collect_all_definitions()

        # Parse and find main_method
        with open(file_path) as f:
            tree = ast.parse(f.read())

        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'main_method':
                func_node = node
                break

        deps = analyzer.get_function_dependencies(func_node, 'myclass.py')

        print(f"  Dependencies of main_method:")
        for dep in deps:
            print(f"    - {dep['name']} ({dep['type']}) in {dep['file_path']}:{dep['line_start']}")

        dep_names = [d['name'] for d in deps]
        has_helper = 'helper' in dep_names

        print()
        print(f"  [{'PASS' if has_helper else 'FAIL'}] Resolved self.helper(): {has_helper}")
        print()

        return has_helper


if __name__ == '__main__':
    print("Running tests for build_dataset.py fixes...\n")

    results = []

    results.append(('Attribute Call Column Offset', test_attribute_call_column_offset()))
    results.append(('Decorator Dependencies', test_decorator_dependencies()))
    results.append(('Jedi Resolution', test_jedi_resolution()))
    results.append(('Method Call Resolution', test_method_call_resolution()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed.")
        sys.exit(1)
