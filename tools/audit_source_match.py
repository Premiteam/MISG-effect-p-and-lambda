import ast
import sys
from pathlib import Path

class RemoveDocstrings(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = self.clean(node.body)
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        node.body = self.clean(node.body)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.body = self.clean(node.body)
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        node.body = self.clean(node.body)
        return node

    @staticmethod
    def clean(body):
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body


def normalized(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ast.dump(RemoveDocstrings().visit(tree), include_attributes=False)


def symbols(path):
    tree = RemoveDocstrings().visit(ast.parse(path.read_text(encoding="utf-8")))
    return {
        getattr(node, "name", f"<module:{index}>"): ast.dump(
            node, include_attributes=False
        )
        for index, node in enumerate(tree.body)
    }


def main():
    draft = Path(sys.argv[1])
    packaged = Path(sys.argv[2])
    for source in sorted(draft.rglob("*.py")):
        relative = source.relative_to(draft)
        excluded = {
            "version12_qa.py",
            "check_final_build.py",
            "render_revision.py",
            "verify_review_delivery.py",
        }
        if (
            any(part.startswith("backup") for part in relative.parts)
            or "daisy_robot_arm" in relative.parts
            or source.name in excluded
        ):
            continue
        target = packaged / relative
        if not target.is_file():
            print("MISSING", relative)
        elif normalized(source) != normalized(target):
            print("DIFFERENT", relative)
            before = symbols(source)
            after = symbols(target)
            print(
                "  ",
                ", ".join(
                    sorted(
                        name
                        for name in before.keys() | after.keys()
                        if before.get(name) != after.get(name)
                    )
                ),
            )


if __name__ == "__main__":
    main()
