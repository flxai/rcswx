"""Static recovery checks; no einsearch/PyTorch dependency or model execution."""
import ast
import hashlib
import json
import unittest

import recover


class RecoveryTests(unittest.TestCase):
    def test_all_model_strings_and_trees(self):
        manifest = json.loads((recover.BASE / "manifest.json").read_text())
        self.assertEqual(len(manifest["models"]), 12)
        for source, names in recover.MODELS.items():
            values, _, _ = recover.read_assignments(recover.SOURCES / source)
            for name in names:
                with self.subTest(name=name):
                    record = manifest["models"][name]
                    value = values[name]
                    self.assertEqual((recover.BASE / record["expanded_definition"]).read_bytes(), value.encode())
                    tokens = recover.loader_tokens(value)
                    self.assertEqual(len(tokens), record["expanded_operation_count"])
                    tree = recover.DSLParser(value).parse()
                    recover.check_arities(tree)
                    self.assertEqual(recover.sha256(recover.canonical(tree)), record["syntax_tree_sha256"])
                    self.assertEqual(recover.sha256(recover.canonical(tokens)), record["ordered_source_operations_sha256"])
                    if name in recover.TARGETS:
                        self.assertEqual(len(tokens), recover.TARGETS[name])

    def test_source_checksums(self):
        manifest = json.loads((recover.BASE / "manifest.json").read_text())
        for source in manifest["sources"].values():
            data = (recover.BASE / source["relative_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), source["sha256"])

    def test_resnet_stage_template_invocations(self):
        module = ast.parse((recover.SOURCES / "resnet.py").read_text())
        for symbol, expected in {"resnet18_no_maxpool": [2, 2, 2, 2], "resnet34_no_maxpool": [3, 4, 6, 3]}.items():
            assignment = next(n for n in module.body if n.targets[0].id == symbol)
            calls = [n for n in ast.walk(assignment.value) if isinstance(n, ast.Call)]
            counts = [sum(ast.literal_eval(n.args[0]) == f"linear{width}" for n in calls) for width in (64, 128, 256, 512)]
            self.assertEqual(counts, expected)

    def test_importable_definitions(self):
        import definitions
        self.assertEqual(len(definitions.MODEL_DEFINITIONS), 12)
        for name, text in definitions.MODEL_DEFINITIONS.items():
            self.assertEqual((recover.BASE / "expanded" / f"{name}.einspace.txt").read_bytes(), text.encode())

    def test_only_supported_source_expressions(self):
        with self.assertRaises(ValueError):
            recover.evaluate(ast.parse("__import__('os').getcwd()", mode="eval").body, {})
        with self.assertRaises(ValueError):
            recover.evaluate(ast.parse("'a' + 'b'", mode="eval").body, {})


if __name__ == "__main__":
    unittest.main()
