import os
from pathlib import Path
from secrets import token_urlsafe
from types import SimpleNamespace

from pydantic import SecretStr
import pytest


ROOT = Path(__file__).resolve().parents[2]
JWT_NAMES = ("JWT_SECRET", "JWT_ISSUER", "JWT_AUDIENCE")


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    from askpolicy import config
    from askpolicy.auth.jwt import AuthenticationError, JWTConfig

    values = {
        "JWT_SECRET": token_urlsafe(48),
        "JWT_ISSUER": "runtime-test-issuer",
        "JWT_AUDIENCE": "runtime-test-api",
    }
    expected = JWTConfig(
        SecretStr(values["JWT_SECRET"]), values["JWT_ISSUER"], values["JWT_AUDIENCE"], "HS256",
    )
    return SimpleNamespace(config=config, error=AuthenticationError, values=values, expected=expected)


def _assert_invalid(runtime, source):
    with pytest.raises(runtime.error) as raised:
        runtime.config.load_jwt_config(source)
    assert raised.value.code == str(raised.value) == "INVALID_JWT_CONFIG"


def test_loader_constructs_validated_jwt_config_from_explicit_mapping(runtime, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "placeholder")
    assert runtime.config.load_jwt_config(runtime.values) == runtime.expected


def test_default_loader_uses_process_environment_without_mutating_it(runtime, monkeypatch):
    for name, value in runtime.values.items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ)
    assert runtime.config.load_jwt_config() == runtime.expected
    unchanged = dict(os.environ) == before
    assert unchanged


@pytest.mark.parametrize("missing", JWT_NAMES)
def test_missing_required_value_fails_with_no_defaults(runtime, monkeypatch, missing):
    for name, value in runtime.values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing)
    _assert_invalid(runtime, None)


@pytest.mark.parametrize("name, value", [
    ("JWT_SECRET", ""),
    ("JWT_SECRET", "short"),
    ("JWT_SECRET", "placeholder"),
    ("JWT_SECRET", "replace-me-with-a-longer-random-key"),
    ("JWT_SECRET", b"not-an-environment-string"),
    ("JWT_ISSUER", ""),
    ("JWT_ISSUER", "placeholder"),
    ("JWT_AUDIENCE", ""),
    ("JWT_AUDIENCE", "placeholder"),
    ("JWT_AUDIENCE", ["wrong-type"]),
], ids=[
    "empty-secret", "short-secret", "placeholder-secret", "long-placeholder-secret", "wrong-secret-type",
    "empty-issuer", "placeholder-issuer", "empty-audience", "placeholder-audience", "wrong-audience-type",
])
def test_invalid_environment_value_uses_existing_jwt_validation(runtime, name, value):
    _assert_invalid(runtime, {**runtime.values, name: value})


def test_hs256_is_fixed_and_unrelated_environment_cannot_supply_missing_credentials(runtime):
    source = {**runtime.values, "JWT_ALGORITHM": "none", "API_PORT": "not-a-port"}
    assert runtime.config.load_jwt_config(source) == runtime.expected
    del source["JWT_SECRET"]
    source["JWT_SIGNING_KEY"] = runtime.values["JWT_SECRET"]
    _assert_invalid(runtime, source)


def test_explicit_empty_mapping_never_falls_back_to_process_environment(runtime, monkeypatch):
    for name, value in runtime.values.items():
        monkeypatch.setenv(name, value)
    _assert_invalid(runtime, {})


def test_nonmapping_input_fails_controlled(runtime):
    _assert_invalid(runtime, [("JWT_SECRET", "not-a-mapping")])


def test_loader_ignores_dotenv_files(runtime, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("".join(f"{name}={value}\n" for name, value in runtime.values.items()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for name in JWT_NAMES:
        monkeypatch.delenv(name, raising=False)
    _assert_invalid(runtime, None)


def test_config_does_not_expose_secret_in_repr_errors_or_logs(runtime, caplog, capsys):
    loaded = runtime.config.load_jwt_config(runtime.values)
    with pytest.raises(runtime.error) as raised:
        runtime.config.load_jwt_config({**runtime.values, "JWT_AUDIENCE": "placeholder"})
    captured = capsys.readouterr()
    emitted = str(raised.value) + caplog.text + captured.out + captured.err
    assert not any(value in emitted for value in runtime.values.values())
    secret_hidden = runtime.values["JWT_SECRET"] not in repr(loaded)
    assert secret_hidden


def test_legacy_settings_fields_and_explicit_values_stay_compatible(runtime):
    fields = set(JWT_NAMES) | {"OLLAMA_HOST", "DB_PATH", "API_PORT", "UI_PORT"}
    assert set(runtime.config.Settings.model_fields) == fields
    assert runtime.config.Settings().model_dump() == dict.fromkeys(fields)
    settings = runtime.config.Settings(
        JWT_SECRET=SecretStr(runtime.values["JWT_SECRET"]),
        JWT_ISSUER=runtime.values["JWT_ISSUER"], JWT_AUDIENCE=runtime.values["JWT_AUDIENCE"],
        DB_PATH=Path("synthetic.sqlite"), OLLAMA_HOST="http://localhost:11434", API_PORT=8000, UI_PORT=8501,
    )
    assert settings.JWT_SECRET == runtime.expected.signing_secret
    assert settings.JWT_ISSUER == runtime.expected.issuer
    assert settings.JWT_AUDIENCE == runtime.expected.audience
    assert settings.DB_PATH == Path("synthetic.sqlite")
    assert (settings.OLLAMA_HOST, settings.API_PORT, settings.UI_PORT) == ("http://localhost:11434", 8000, 8501)
