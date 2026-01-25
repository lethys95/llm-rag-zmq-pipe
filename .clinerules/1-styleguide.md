1. When writing print statements, if you need to create multiple lines, prefer multiline strings with dedent from textwrap instead of multiple print statements.

2. Do not write arbitrary comments.

3. Prefer union types instead of types from typing. After 3.10, many use cases of the typing package has been replaced. So prefer union types over optionals, e.g. str | None over Optional[str], etc. Other redundant types would be Union, List, Dict which also have native support now.

4. Do NOT use imports inside functions or classes unless there's a very good reason to do so. A good reason would be if the dependency is an optional feature and there needs to be a check. A bad reason is lazines or feeling scared that something might break. If an import would break the system, then there's an architecture issue, and we have bigger problems.

5. Disallow any type. Create dataclasses instead. This includes dict[str, Any]. I don't want to see them. use dataclasses.

6. Do not use if TYPE_CHECKING from the typing package. If there's a circular import, then I want to know instead of pushing the problem underneath the rug.

7. Do not use string types. Use real imports.

8. Target Python 3.11+.

9. Write PROFESSIONAL code.

10. Use absolute, not relative imports.

11. If you get import errors, make sure you're in the environment. So, source .venv/bin/activate. If you still get import errors, then you can assume the package is missing. Make sure you otherwise add it with uv.