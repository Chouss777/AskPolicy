import ast
import runpy
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "askpolicy" / "config.py"
SETTING_NAMES = (
    "JWT_SECRET",
    "JWT_ISSUER",
    "JWT_AUDIENCE",
    "OLLAMA_HOST",
    "DB_PATH",
    "API_PORT",
    "UI_PORT",
)
DEFERRED_COMMENT = """# DEFERRED (Phase 3): environment loading is not implemented in
# Phase 1. Pydantic v2 moved BaseSettings into the separate
# pydantic-settings package, which is not in the locked stack
# (docs/decisions.md §2). When Phase 3 requires env loading from
# .env, evaluate whether pydantic-settings justifies a stack
# amendment under docs/decisions.md §18. Do not add it without
# following that procedure."""


def test_settings_loads_with_defaults(monkeypatch):
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)

    settings_class = runpy.run_path(str(CONFIG_PATH))["Settings"]
    settings = settings_class()

    assert set(settings_class.model_fields) == set(SETTING_NAMES)
    assert settings.model_dump() == dict.fromkeys(SETTING_NAMES)


def test_env_loading_deferred_is_documented():
    source = CONFIG_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    settings_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
    )
    comment_lines = DEFERRED_COMMENT.splitlines()
    class_line = settings_class.lineno - 1

    assert source.splitlines()[class_line - len(comment_lines):class_line] == comment_lines
    assert [ast.unparse(base) for base in settings_class.bases] == ["BaseModel"]

    prohibited_imports = {"BaseSettings", "pydantic_settings"}
    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported_names = {alias.name.split(".")[0] for alias in node.names}
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module.split(".")[0])
            assert imported_names.isdisjoint(prohibited_imports)
