from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "askpolicy"
REQUIRED_SUBPACKAGES = (
    "api",
    "auth",
    "ingestion",
    "retrieval",
    "generation",
    "security",
    "storage",
    "observability",
)


def test_main_package_exists():
    assert PACKAGE_ROOT.is_dir(), f"Missing package directory: {PACKAGE_ROOT}"
    assert (PACKAGE_ROOT / "__init__.py").is_file(), f"Missing initializer in {PACKAGE_ROOT}"


@pytest.mark.parametrize("subpackage", REQUIRED_SUBPACKAGES)
def test_subpackage_exists_under_package_root(subpackage):
    package_directory = PACKAGE_ROOT / subpackage
    assert package_directory.is_dir(), f"Missing package directory: {package_directory}"
    assert (package_directory / "__init__.py").is_file(), (
        f"Missing initializer in {package_directory}"
    )
