import os
from unittest.mock import patch

from episodiq.api_adapters import OpenAIConfig


class TestAdapterConfig:
    def test_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            config = OpenAIConfig()
            assert config.id == "openai"
            assert config.upstream_base_url == "https://api.openai.com/v1"
            assert config.timeout == 120.0

    def test_from_env(self):
        with patch.dict(
            os.environ,
            {"EPISODIQ_OPENAI_BASE_URL": "https://custom.openai.com"},
        ):
            config = OpenAIConfig()
            assert config.upstream_base_url == "https://custom.openai.com"

    def test_override_defaults(self):
        config = OpenAIConfig(
            id="custom-id",
            upstream_base_url="https://override.com",
            timeout=60.0,
        )
        assert config.id == "custom-id"
        assert config.upstream_base_url == "https://override.com"
        assert config.timeout == 60.0
