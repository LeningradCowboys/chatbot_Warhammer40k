from pathlib import Path

import pytest
from pydantic import ValidationError

from src.rag.config import Settings


def _key_error_fields(error: ValidationError) -> set[tuple]:
    return {tuple(detail["loc"]) for detail in error.errors()}


@pytest.fixture(autouse=True)
def no_key_in_environment(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_missing_key_is_rejected():
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    assert _key_error_fields(error.value) == {("google_api_key",)}


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_key_is_rejected_like_a_missing_one(monkeypatch, value):
    monkeypatch.setenv("GOOGLE_API_KEY", value)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    assert _key_error_fields(error.value) == {("google_api_key",)}


def test_set_key_is_kept_without_surrounding_spaces(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "  abc123  ")
    assert Settings(_env_file=None).google_api_key.get_secret_value() == "abc123"


def test_key_is_masked_when_printed(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    assert "abc123" not in repr(Settings(_env_file=None))


def test_chat_temperature_has_one_validated_environment_setting(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    assert Settings(_env_file=None).chat_model == "gemini-3.8-flash"
    assert Settings(_env_file=None).chat_temperature is None

    monkeypatch.setenv("CHAT_TEMPERATURE", "0")
    assert Settings(_env_file=None).chat_temperature == 0.0


@pytest.mark.parametrize("value", ["-0.1", "2.1"])
def test_chat_temperature_rejects_values_outside_the_api_range(monkeypatch, value):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    monkeypatch.setenv("CHAT_TEMPERATURE", value)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    assert _key_error_fields(error.value) == {("chat_temperature",)}


def test_env_file_is_read_only_when_given(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=from-file\n", encoding="utf-8")
    assert Settings(_env_file=env_file).google_api_key.get_secret_value() == "from-file"
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
