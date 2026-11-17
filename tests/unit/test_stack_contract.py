import tomllib
import unittest
from pathlib import Path


PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


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
                "torch==2.15.0",
                "transformers==5.23.0",
                "uvicorn",
            ],
        )
        self.assertFalse(project.get("optional-dependencies"))

    def test_torch_uses_only_the_explicit_cpu_index(self):
        uv = self.pyproject["tool"]["uv"]
        self.assertEqual(uv["sources"], {"torch": {"index": "pytorch-cpu"}})
        self.assertEqual(uv["index"], [{"name": "pytorch-cpu", "url": CPU_INDEX, "explicit": True}])

    def test_lock_records_approved_sources_and_cpu_wheels(self):
        with PYPROJECT_PATH.with_name("uv.lock").open("rb") as lock_file:
            packages = tomllib.load(lock_file)["package"]
        torch = [package for package in packages if package["name"] == "torch"]
        self.assertCountEqual([package["version"] for package in torch], ["2.15.0", "2.15.0+cpu"])
        self.assertTrue(all(package["source"] == {"registry": CPU_INDEX} for package in torch))
        cpu = next(package for package in torch if package["version"] == "2.15.0+cpu")
        for platform in ("win_amd64", "manylinux_2_28_x86_64"):
            matches = [wheel for wheel in cpu["wheels"] if wheel["url"].endswith(f"-cp311-cp311-{platform}.whl")]
            self.assertEqual(len(matches), 1)
            self.assertRegex(matches[0]["hash"], r"^sha256:[0-9a-f]{64}$")
        transformers = [package for package in packages if package["name"] == "transformers"]
        self.assertEqual(len(transformers), 1)
        self.assertEqual(transformers[0]["version"], "5.23.0")
        self.assertEqual(transformers[0]["source"], {"registry": "https://pypi.org/simple"})
        for package in packages:
            if package["name"] != "torch":
                self.assertNotEqual(package["source"].get("registry"), CPU_INDEX)
            self.assertFalse(package["name"].startswith(("nvidia-", "cuda-", "pytorch-triton")))
            self.assertNotIn(package["name"], {"sentence-transformers", "onnxruntime", "triton"})

    def test_pytest_is_the_development_dependency(self):
        self.assertEqual(self.pyproject["dependency-groups"], {"dev": ["pytest"]})
