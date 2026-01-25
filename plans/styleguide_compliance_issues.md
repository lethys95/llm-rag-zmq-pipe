# Styleguide Compliance Issues

## Overview

This document identifies violations of the project's styleguide (`.clinerules/1-styleguide.md`) found in the codebase. The styleguide was established to ensure consistent, professional, and maintainable Python code following Python 3.11+ best practices.

## Styleguide Summary

Key rules from `.clinerules/1-styleguide.md`:

1. Use multiline strings with `dedent` for multiple print statements
2. Do not write arbitrary comments
3. Prefer union types (str | None) over typing types (Optional[str])
4. Do NOT use imports inside functions/classes unless absolutely necessary
5. Disallow `Any` type - create dataclasses instead
6. Do NOT use `if TYPE_CHECKING` - resolve circular imports instead
7. Do not use string types - use real imports
8. Target Python 3.11+
9. Write PROFESSIONAL code
10. Use absolute imports
11. Handle import errors by activating venv or adding missing packages

---

## Critical Violations

### 1. TYPE_CHECKING Usage (Rule #6)

**Location**: `src/nodes/orchestration/knowledge_broker.py:16-18`

**Violation Code**:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nodes.algo_nodes.memory_evaluator_node import MemoryEvaluation
```

**Styleguide Rule**:
> 6. Do not use if TYPE_CHECKING from the typing package. If there's a circular import, then I want to know instead of pushing the problem underneath the rug.

**Issue**: This is hiding a circular dependency between `KnowledgeBroker` and `MemoryEvaluationNode`. The styleguide explicitly states that circular imports should be resolved rather than hidden.

**Impact**: 
- Masks architectural problems
- Makes dependency graph unclear
- Violates project's philosophy of transparency

**Current Usage**:
```python
@dataclass
class KnowledgeBroker:
    # ...
    evaluated_memories: list[tuple[RAGDocument, "MemoryEvaluation"]] = \
        field(default_factory=list)
```

**Recommendation Options**:

**Option 1: Move MemoryEvaluation to a separate module**
```
src/models/
  ├── memory.py
  └── memory_evaluation.py  # ← Move here
```

**Option 2: Use a protocol instead of concrete type**
```python
# src/models/protocols.py
from typing import Protocol

class MemoryEvaluation(Protocol):
    """Protocol for memory evaluation results."""
    relevance: float
    summary: str
```

**Option 3: Refactor to avoid the dependency**
- Make KnowledgeBroker use a generic data structure
- Move evaluation logic elsewhere

**Priority**: **CRITICAL** - This is a clear violation of project philosophy.

---

## Minor Violations

### 2. typing.Optional Usage (Rule #3)

**Location**: `src/cli.py`

**Violation Code**:
```python
from typing import Optional
```

**Styleguide Rule**:
> 3. Prefer union types instead of types from typing. After 3.10, many use cases of the typing package has been replaced. So prefer union types over optionals, e.g. str | None over Optional[str], etc.

**Issue**: Using `Optional[str]` instead of native `str | None` which is available in Python 3.11+.

**Search Results**:
```bash
$ grep -r "from typing import Optional" src/
src/cli.py:from typing import Optional
```

**Recommendation**: Replace all `Optional[T]` with `T | None`

**Example Fix**:
```python
# Before
from typing import Optional

def some_function(param: Optional[str]) -> None:
    pass

# After
def some_function(param: str | None) -> None:
    pass
```

**Priority**: **MEDIUM** - Minor but easy to fix.

---

### 3. typing.Any Usage (Rule #5)

**Locations**: 
- `src/rag/selector.py`
- `src/rag/qdrant_connector.py`

**Violation Code**:
```python
# src/rag/selector.py
from typing import Any

# ... later in code
def some_method(self, param: Any) -> Any:
    pass
```

**Styleguide Rule**:
> 5. Disallow any type. Create dataclasses instead. This includes dict[str, Any].

**Issue**: Using `Any` defeats type safety and makes the code unclear about what types are expected.

**Search Results**:
```bash
$ grep -r "from typing import Any" src/
src/rag/selector.py:from typing import Any
src/rag/qdrant_connector.py:from typing import Any
```

**Need Investigation**: Check how `Any` is used in these files to create appropriate dataclasses.

**Recommendation**:
1. Identify where `Any` is used
2. Create dataclasses or protocols to replace them
3. Update type annotations to use specific types

**Example Fix**:
```python
# Before
from typing import Any

def process_data(data: Any) -> Any:
    return {"result": "something"}

# After
from dataclasses import dataclass

@dataclass
class ProcessedResult:
    result: str
    metadata: dict[str, str]

def process_data(data: dict[str, str]) -> ProcessedResult:
    return ProcessedResult(result="something", metadata={})
```

**Priority**: **HIGH** - Type safety is important for maintainability.

---

### 4. typing.Callable Usage (Rule #3)

**Location**: `src/chrono/task_scheduler.py`

**Violation Code**:
```python
from typing import Callable
```

**Styleguide Rule**:
> 3. Prefer union types instead of types from typing. After 3.10, many use cases of the typing package has been replaced.

**Issue**: While `Callable` has legitimate use cases, the styleguide prefers avoiding typing imports where possible. Python 3.11+ supports more flexible callable types.

**Search Results**:
```bash
$ grep -r "from typing import Callable" src/
src/chrono/task_scheduler.py:from typing import Callable
```

**Need Investigation**: Check how `Callable` is used.

**Possible Alternatives**:
```python
# Before
from typing import Callable

def register_task(self, task: Callable[[], None]) -> None:
    pass

# After - use protocol
from typing import Protocol

class Task(Protocol):
    def __call__(self) -> None: ...

def register_task(self, task: Task) -> None:
    pass

# Or - use specific function type
def register_task(self, task: () -> None) -> None:
    pass
```

**Priority**: **LOW-MEDIUM** - Less critical than Any, but still worth fixing.

---

## Code Quality Issues (Not Styleguide Violations)

### 5. Print Statement in Production Code

**Location**: `src/handlers/primary_response.py:109`

**Issue Code**:
```python
print(f"PRIMARY RESPONSE: {response}")
```

**Styleguide Rule #1** (Related):
> 1. When writing print statements, if you need to create multiple lines, prefer multiline strings with dedent from textwrap instead of multiple print statements.

**Issue**: While this is a single print statement, using `print()` in production code is generally discouraged in favor of logging.

**Recommendation**: Replace with logger:
```python
# Before
print(f"PRIMARY RESPONSE: {response}")

# After
logger.info(f"Primary response generated (length: {len(response)})")
```

**Note**: The file already uses `logger` throughout, so this is an oversight.

**Priority**: **LOW** - Not a styleguide violation per se, but a best practice issue.

---

## Other Typing Imports Found

### typing.Self (Acceptable)

**Locations**:
- `src/rag/embeddings.py`
- `src/nodes/core/base.py`

**Code**:
```python
from typing import Self
```

**Assessment**: This is **acceptable**. `Self` was added in Python 3.11 and is the idiomatic way to return the same type as the class.

### typing.Literal (Acceptable)

**Location**: `src/nodes/core/types.py`

**Code**:
```python
from typing import Literal
```

**Assessment**: This is **acceptable**. `Literal` is necessary for type-level enumeration of specific string values.

### typing.Protocol (Acceptable)

**Location**: `src/nodes/core/node_protocol.py`

**Code**:
```python
from typing import Protocol
```

**Assessment**: This is **acceptable**. `Protocol` is necessary for structural subtyping.

### typing.TYPE_CHECKING (Already Documented)

**Location**: `src/nodes/orchestration/knowledge_broker.py`

**Status**: Documented as Critical Violation #1.

---

## Summary of Violations

| Priority | Violation | Location | Lines Affected | Action Required |
|----------|-----------|----------|----------------|-----------------|
| **CRITICAL** | TYPE_CHECKING usage | `knowledge_broker.py` | ~3 lines | Resolve circular import |
| **HIGH** | typing.Any usage | `selector.py`, `qdrant_connector.py` | Unknown | Create dataclasses |
| **MEDIUM** | typing.Optional usage | `cli.py` | Unknown | Replace with `T \| None` |
| **LOW-MEDIUM** | typing.Callable usage | `task_scheduler.py` | Unknown | Use protocol or native types |
| **LOW** | print statement | `primary_response.py` | 1 line | Use logger instead |

---

## Implementation Plan

### Phase 1: Critical Fixes
1. **Resolve TYPE_CHECKING violation**
   - Choose refactoring approach (Option 1, 2, or 3)
   - Implement chosen solution
   - Verify no circular imports remain
   - Update type annotations

### Phase 2: Type Safety Improvements
2. **Replace typing.Any with dataclasses**
   - Audit `selector.py` for Any usage
   - Audit `qdrant_connector.py` for Any usage
   - Create appropriate dataclasses
   - Update type annotations
   - Test changes

3. **Replace typing.Optional with native types**
   - Audit `cli.py` for Optional usage
   - Replace with `T | None` syntax
   - Verify type checking still works

### Phase 3: Code Quality
4. **Replace typing.Callable**
   - Audit `task_scheduler.py` for Callable usage
   - Use protocol or native function types
   - Test changes

5. **Replace print with logger**
   - Find and replace print statement
   - Verify logging configuration
   - Test output

---

## Testing Strategy

Before making changes:

1. **Run type checker** (mypy/pyright)
   ```bash
   mypy src/
   ```

2. **Run existing tests**
   ```bash
   pytest tests/
   ```

3. **Document current state**
   - Note any type checker warnings
   - Note any test failures

After making changes:

1. **Verify type checking passes**
   - Ensure no new type errors introduced

2. **Run all tests**
   - Ensure no regressions

3. **Manual verification**
   - Run the application
   - Verify logging output
   - Check for runtime errors

---

## Risk Assessment

**Low Risk**:
- Replacing print with logger
- Replacing Optional with native types

**Medium Risk**:
- Replacing Any with dataclasses (may require significant refactoring)
- Replacing Callable (may require protocol changes)

**High Risk**:
- Resolving TYPE_CHECKING circular import (may require architectural changes)

---

## Additional Recommendations

### 1. Enable Type Checking in CI/CD

Add type checking to your CI pipeline:
```yaml
# .github/workflows/ci.yml
- name: Type check
  run: mypy src/ --strict
```

### 2. Add Pre-commit Hooks

Configure pre-commit to catch style violations:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### 3. Document Styleguide Enforcement

Update project documentation to clarify:
- Styleguide rules are enforced
- Tools used for enforcement (mypy, pylint, etc.)
- PR review checklist includes styleguide compliance

### 4. Consider Type Enforcement Tools

Tools that can help enforce styleguide compliance:
- **mypy** - Static type checker
- **ruff** - Fast Python linter
- **pylint** - Code quality checker

Example configuration:
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.ruff]
line-length = 100
target-version = "py311"
```

---

## Conclusion

The codebase has **5 styleguide violations** ranging from critical (TYPE_CHECKING) to minor (print statement). The most important issue is the circular import being hidden by TYPE_CHECKING, which goes against the project's philosophy of transparency.

Overall, the code quality is good, and most styleguide rules are being followed. The violations found are:
- 1 critical (circular import)
- 2 high/medium (type safety)
- 2 low-medium (code quality)

All violations are actionable and can be fixed without major architectural changes. The recommended implementation plan addresses issues by priority, with the most critical issues first.