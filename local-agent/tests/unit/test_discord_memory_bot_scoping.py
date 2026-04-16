"""
Regression tests for name-scoping bugs in agent/discord_memory_bot.py.

The on_message handler is long and calls several helpers (get_recent_summaries,
buffer_for_summary, maybe_generate_summaries, ...) that are imported at module
scope. A local `from .conversation_context import ...` inside on_message would
make Python treat the name as function-local everywhere in the function, so any
reference reached before that local import raises UnboundLocalError (see
commit 61b9415 for the original incident and TK-471 for the re-filed crash).

These tests walk the AST of discord_memory_bot.py to guarantee the names stay
module-scoped so the runtime bug cannot silently regress.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "agent" / "discord_memory_bot.py"
)

# Names imported at module level from .conversation_context. A local re-import
# inside on_message shadows the module name and reintroduces the crash.
CONVERSATION_CONTEXT_NAMES = {
    "buffer_for_summary",
    "get_conversation_context_tools",
    "get_recent_summaries",
    "maybe_generate_summaries",
}


def _parse_module() -> ast.Module:
    source = MODULE_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(MODULE_PATH))


def _find_function(tree: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found in {MODULE_PATH}")


def _collect_local_bindings(fn: ast.AST) -> dict[str, int]:
    """Return {name: lineno} for every binding the Python compiler would treat
    as function-local inside ``fn`` — i.e. assignments and import statements
    at any depth that is not a nested function/class body.
    """
    bindings: dict[str, int] = {}

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            # Nested functions/classes have their own scope — skip.
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    bindings.setdefault(alias.asname or alias.name, child.lineno)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    top = (alias.asname or alias.name).split(".")[0]
                    bindings.setdefault(top, child.lineno)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    for n in _iter_assign_names(target):
                        bindings.setdefault(n, child.lineno)
            elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
                if isinstance(child.target, ast.Name):
                    bindings.setdefault(child.target.id, child.lineno)
            walk(child)

    walk(fn)
    return bindings


def _iter_assign_names(target: ast.AST):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _iter_assign_names(elt)


@pytest.fixture(scope="module")
def on_message_bindings() -> dict[str, int]:
    tree = _parse_module()
    fn = _find_function(tree, "on_message")
    return _collect_local_bindings(fn)


def test_on_message_does_not_shadow_get_recent_summaries(on_message_bindings: dict[str, int]) -> None:
    """TK-471 regression: a local import of get_recent_summaries inside
    on_message makes the name function-local and raises UnboundLocalError at
    the first use. The only binding must come from the top-of-file import.
    """
    assert "get_recent_summaries" not in on_message_bindings, (
        "on_message locally binds get_recent_summaries at line "
        f"{on_message_bindings.get('get_recent_summaries')!r}, which shadows "
        "the module-level import and re-introduces TK-471's UnboundLocalError."
    )


def test_on_message_does_not_shadow_conversation_context_imports(
    on_message_bindings: dict[str, int],
) -> None:
    """Same class of bug for every name imported from conversation_context."""
    shadowed = {
        name: on_message_bindings[name]
        for name in CONVERSATION_CONTEXT_NAMES
        if name in on_message_bindings
    }
    assert not shadowed, (
        "on_message locally binds names that are already imported at module "
        f"level: {shadowed}. Remove the local `from .conversation_context "
        "import ...` statement — a local binding turns the name function-local "
        "and every earlier reference raises UnboundLocalError."
    )


def test_conversation_context_names_are_module_level_imports() -> None:
    """Sanity-check the premise: the names really are imported at module scope."""
    tree = _parse_module()
    module_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "conversation_context":
            module_imports.update(alias.asname or alias.name for alias in node.names)
        # also match relative `from .conversation_context import ...`
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("conversation_context"):
            module_imports.update(alias.asname or alias.name for alias in node.names)

    missing = CONVERSATION_CONTEXT_NAMES - module_imports
    assert not missing, (
        f"Expected module-level import of {missing} from conversation_context "
        "but none was found. Update CONVERSATION_CONTEXT_NAMES or restore the "
        "module-level import."
    )
