import ast
import io
import sys
import tokenize
from pathlib import Path


def strip_file(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    locations = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            locations.append((token.start[0] - 1, token.start[1], token.end[1]))
    for line_index, start, end in reversed(locations):
        line = lines[line_index]
        lines[line_index] = line[:start] + line[end:]
    tree = ast.parse("".join(lines), filename=str(path))
    spans = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
            continue
        if not isinstance(first.value.value, str):
            continue
        spans.append(
            (
                first.lineno - 1,
                first.end_lineno,
                len(node.body) == 1 and not isinstance(node, ast.Module),
            )
        )
    for start, end, needs_pass in sorted(spans, reverse=True):
        indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
        for index in range(start, end):
            lines[index] = "\n" if lines[index].endswith("\n") else ""
        if needs_pass:
            lines[start] = indent + "pass\n"
    path.write_text("".join(lines), encoding="utf-8")
    return len(locations) + len(spans)


def main() -> None:
    root = Path(sys.argv[1])
    files = sorted(root.rglob("*.py"))
    removed = sum(strip_file(path) for path in files)
    print(f"Processed {len(files)} Python files; removed {removed} comment tokens")


if __name__ == "__main__":
    main()
