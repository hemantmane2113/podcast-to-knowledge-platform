import pytest

from app.config.settings import Settings


def test_development_boots_without_secrets() -> None:
    settings = Settings(_env_file=None, app_env="development")
    assert settings.supadata_api_key == ""


def test_non_development_requires_secrets() -> None:
    with pytest.raises(ValueError, match="SUPADATA_API_KEY"):
        Settings(_env_file=None, app_env="production")


def test_non_development_boots_when_secrets_present() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        supadata_api_key="key",
        llm_api_key="key",
        admin_auth_secret="secret",
    )
    assert settings.app_env == "production"
