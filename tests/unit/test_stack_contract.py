import tomllib
import unittest
from pathlib import Path


PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


class StackContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with PYPROJECT_PATH.open("rb") as pyproject_file:
            cls.pyproject = tomllib.load(pyproject_file)

    def test_pyproject_is_valid_toml_with_a_project_table(self):
        self.assertIsInstance(self.pyproject["project"], dict)

    def test_project_name(self):
        self.assertEqual(self.pyproject["project"]["name"], "askpolicy")

    def test_project_description(self):
        self.assertEqual(
            self.pyproject["project"]["description"],
            "Secure multilingual enterprise RAG",
        )

    def test_python_requirement(self):
        self.assertEqual(self.pyproject["project"]["requires-python"], ">=3.11,<3.12")

    def test_runtime_dependencies_match_locked_stack(self):
        project = self.pyproject["project"]
        self.assertCountEqual(
            project["dependencies"],
            [
                "bm25s",
                "faiss-cpu",
                "fastapi",
                "pwdlib[argon2]",
                "pydantic>=2,<3",
                "PyJWT",
                "streamlit",
                "uvicorn",
            ],
        )
        self.assertFalse(project.get("optional-dependencies"))

    def test_pytest_is_the_development_dependency(self):
        self.assertEqual(self.pyproject["dependency-groups"], {"dev": ["pytest"]})
