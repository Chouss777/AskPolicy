from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPOSITORY_ROOT / "tests"
TEST_PACKAGES = ("integration", "security", "eval")
REQUIRED_CONTRACT_FILES = (
    "AGENTS.md",
    "docs/decisions.md",
    "docs/codex-checklist.md",
    "pyproject.toml",
    "uv.lock",
)


@pytest.mark.parametrize("name", ("unit", *TEST_PACKAGES))
def test_required_test_directory_exists(name):
    directory = TESTS_ROOT / name
    assert directory.is_dir(), f"Missing test directory: {directory}"


@pytest.mark.parametrize("name", TEST_PACKAGES)
def test_test_package_has_initializer(name):
    initializer = TESTS_ROOT / name / "__init__.py"
    assert initializer.is_file(), f"Missing test package initializer: {initializer}"


@pytest.mark.parametrize("relative_path", REQUIRED_CONTRACT_FILES)
def test_required_contract_file_exists(relative_path):
    contract_file = REPOSITORY_ROOT / relative_path
    assert contract_file.is_file(), f"Missing required repository file: {relative_path}"
