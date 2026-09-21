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
        groq_api_key="key",
        admin_auth_secret="secret",
    )
    assert settings.app_env == "production"


# --- LLM provider key requirement (only the selected provider's key) --------------


def test_non_development_requires_the_selected_providers_key_only() -> None:
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        Settings(
            _env_file=None,
            app_env="production",
            supadata_api_key="key",
            admin_auth_secret="secret",
            llm_provider="groq",
        )


def test_non_development_does_not_require_groq_key_when_openai_selected() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        supadata_api_key="key",
        admin_auth_secret="secret",
        llm_provider="openai",
        openai_api_key="key",
    )
    assert settings.groq_api_key == ""


def test_non_development_does_not_require_openai_key_when_groq_selected() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        supadata_api_key="key",
        admin_auth_secret="secret",
        llm_provider="groq",
        groq_api_key="key",
    )
    assert settings.openai_api_key == ""


def test_opensource_provider_requires_base_url_not_a_key() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        supadata_api_key="key",
        admin_auth_secret="secret",
        llm_provider="opensource",
        opensource_base_url="http://localhost:8080/v1",
    )
    assert settings.opensource_api_key == ""


def test_unknown_llm_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings(
            _env_file=None,
            app_env="production",
            supadata_api_key="key",
            admin_auth_secret="secret",
            llm_provider="not-a-real-provider",
        )
