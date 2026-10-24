import ast
import io
import sys
import tokenize
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_SOURCE = ROOT / "src" / "core"
sys.path.insert(0, str(SIMULATION_SOURCE))

from systems import load_family


class PackageTests(unittest.TestCase):
    def test_frozen_family(self):
        path = ROOT / "data" / "systems.npz"
        family = load_family(path)
        self.assertEqual(len(family["test"]), 201)
        self.assertEqual(len(family["train"]), 201)
        self.assertEqual(len(family["val"]), 102)

    def test_analysis_inputs(self):
        self.assertTrue((ROOT / "data/balanced_full_factorial_results.json").is_file())
        self.assertTrue((ROOT / "data/figure4_source_data.csv").is_file())
        self.assertTrue((ROOT / "data/heating_principal_contrasts.csv").is_file())

    def test_paper_pipeline_inputs(self):
        base = ROOT / "src" / "experiments"
        required = [
            "run_p_lambda_interaction_AUTO.py",
            "run_final_frozen_comparison_AUTO.py",
            "run_new_system_holdout_AUTO.py",
            "run_full_train_gamma_audit.py",
            "run_figure3_local_refinement_201.py",
            "benchmark_p_cost.py",
        ]
        for relative in required:
            self.assertTrue((base / relative).is_file(), relative)

    def test_measured_raw_record_not_bundled(self):
        self.assertFalse((ROOT / "data" / "heating_system.dat.gz").exists())
        self.assertFalse((ROOT / "data" / "test_predictions.csv").exists())

    def test_shallow_layout(self):
        self.assertEqual(
            sorted(
                path.name
                for path in ROOT.iterdir()
                if path.is_dir() and path.name != ".git"
            ),
            ["data", "src", "tests", "tools"],
        )
        allowed = {"analysis", "core", "experiments", "figures", "theory"}
        self.assertEqual(
            {path.name for path in (ROOT / "src").iterdir() if path.is_dir()}, allowed
        )
        self.assertFalse(
            any(
                path.is_dir() and path.name != "__pycache__"
                for path in (ROOT / "data").iterdir()
            )
        )
        self.assertFalse(
            any(
                path.is_dir() and path.name != "__pycache__"
                for path in (ROOT / "tests").iterdir()
            )
        )
        self.assertFalse(
            any(
                path.is_dir() and path.name != "__pycache__"
                for path in (ROOT / "tools").iterdir()
            )
        )

    def test_source_has_no_annotations(self):
        for path in sorted(ROOT.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            comments = [
                token
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            ]
            self.assertEqual(comments, [], str(path))
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    self.assertIsNone(ast.get_docstring(node), str(path))


if __name__ == "__main__":
    unittest.main()
