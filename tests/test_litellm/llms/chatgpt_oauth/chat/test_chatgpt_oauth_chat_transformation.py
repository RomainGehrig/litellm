"""
Tests for ChatGPT OAuth Chat Completion transformation.

Tests the ChatGPTOAuthChatConfig class that handles ChatGPT OAuth-specific
transformations for the Chat Completions API.

Source: litellm/llms/chatgpt_oauth/chat/transformation.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.chatgpt_oauth.chat.transformation import ChatGPTOAuthChatConfig
from litellm.llms.chatgpt_oauth.common_utils import (
    ChatGPTBackendMode,
    ChatGPTOAuthError,
    ChatGPTOAuthTokenManager,
)


class TestChatGPTOAuthChatTransformation:
    """Test ChatGPT OAuth Chat API configuration and transformations"""

    def setup_method(self):
        """Reset singleton before each test"""
        ChatGPTOAuthTokenManager._instance = None

    def test_provider_config_registration(self):
        """Test that ChatGPT OAuth provider returns ChatGPTOAuthChatConfig"""
        config = ProviderConfigManager.get_provider_chat_config(
            model="gpt-4",
            provider=LlmProviders.CHATGPT_OAUTH,
        )

        assert config is not None, "Config should not be None for ChatGPT OAuth provider"
        assert isinstance(
            config, ChatGPTOAuthChatConfig
        ), f"Expected ChatGPTOAuthChatConfig, got {type(config)}"

    def test_custom_llm_provider_property(self):
        """Test that custom_llm_provider returns correct value"""
        config = ChatGPTOAuthChatConfig()

        assert config.custom_llm_provider == "chatgpt_oauth"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_credentials"
    )
    def test_get_openai_compatible_provider_info(self, mock_get_creds):
        """Test _get_openai_compatible_provider_info returns OAuth credentials"""
        mock_get_creds.return_value = ("https://api.openai.com/v1", "sk-oauth-token")

        config = ChatGPTOAuthChatConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )

        assert api_base == "https://api.openai.com/v1"
        assert api_key == "sk-oauth-token"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_credentials"
    )
    def test_get_openai_compatible_provider_info_with_override(self, mock_get_creds):
        """Test _get_openai_compatible_provider_info respects api_base override"""
        mock_get_creds.return_value = ("https://api.openai.com/v1", "sk-oauth-token")

        config = ChatGPTOAuthChatConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base="https://custom.api.com/v1", api_key=None
        )

        # Should use provided api_base but OAuth token
        assert api_base == "https://custom.api.com/v1"
        assert api_key == "sk-oauth-token"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment(self, mock_get_headers):
        """Test validate_environment sets correct headers"""
        mock_get_headers.return_value = {
            "Authorization": "Bearer sk-oauth-token",
            "Content-Type": "application/json",
        }

        config = ChatGPTOAuthChatConfig()
        headers = config.validate_environment(
            headers={},
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer sk-oauth-token"
        assert headers["Content-Type"] == "application/json"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment_preserves_existing_headers(self, mock_get_headers):
        """Test validate_environment preserves existing headers"""
        mock_get_headers.return_value = {
            "Authorization": "Bearer sk-oauth-token",
            "Content-Type": "application/json",
        }

        config = ChatGPTOAuthChatConfig()
        headers = config.validate_environment(
            headers={"X-Custom-Header": "custom-value"},
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["X-Custom-Header"] == "custom-value"
        assert headers["Authorization"] == "Bearer sk-oauth-token"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_api_base"
    )
    def test_get_complete_url_default(self, mock_get_base):
        """Test get_complete_url returns correct endpoint"""
        mock_get_base.return_value = "https://api.openai.com/v1"

        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gpt-4",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://api.openai.com/v1/chat/completions"

    def test_get_complete_url_with_custom_base(self):
        """Test get_complete_url with custom api_base"""
        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base="https://custom.api.com/v1",
            api_key=None,
            model="gpt-4",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://custom.api.com/v1/chat/completions"

    def test_get_complete_url_handles_trailing_slash(self):
        """Test get_complete_url handles trailing slash in api_base"""
        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base="https://api.openai.com/v1/",
            api_key=None,
            model="gpt-4",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://api.openai.com/v1/chat/completions"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_credentials"
    )
    def test_get_api_key_returns_oauth_token(self, mock_get_creds):
        """Test get_api_key returns OAuth token"""
        mock_get_creds.return_value = ("https://api.openai.com/v1", "sk-oauth-token")

        token = ChatGPTOAuthChatConfig.get_api_key()

        assert token == "sk-oauth-token"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_credentials"
    )
    def test_get_api_key_returns_none_on_error(self, mock_get_creds):
        """Test get_api_key returns None when OAuth fails"""
        mock_get_creds.side_effect = ChatGPTOAuthError(401, "No token")

        token = ChatGPTOAuthChatConfig.get_api_key()

        assert token is None

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_api_base"
    )
    def test_get_api_base_default(self, mock_get_base):
        """Test get_api_base returns OAuth api_base"""
        mock_get_base.return_value = "https://api.openai.com/v1"

        api_base = ChatGPTOAuthChatConfig.get_api_base()

        assert api_base == "https://api.openai.com/v1"

    def test_get_api_base_with_override(self):
        """Test get_api_base respects override"""
        api_base = ChatGPTOAuthChatConfig.get_api_base(
            api_base="https://custom.api.com/v1"
        )

        assert api_base == "https://custom.api.com/v1"

    def test_inherits_from_openai_gpt_config(self):
        """Test that ChatGPTOAuthChatConfig inherits OpenAI GPT config methods"""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        config = ChatGPTOAuthChatConfig()

        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params")
        assert hasattr(config, "map_openai_params")
        assert hasattr(config, "transform_request")

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns expected parameters"""
        config = ChatGPTOAuthChatConfig()

        supported = config.get_supported_openai_params("gpt-4")

        # Should include standard OpenAI parameters
        expected_params = [
            "temperature",
            "max_tokens",
            "stream",
            "tools",
            "tool_choice",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]

        for param in expected_params:
            assert param in supported, f"{param} should be in supported params"


class TestChatGPTOAuthChatValidation:
    """Test validation and error handling for Chat API"""

    def setup_method(self):
        """Reset singleton before each test"""
        ChatGPTOAuthTokenManager._instance = None

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment_raises_on_oauth_error(self, mock_get_headers):
        """Test validate_environment raises when OAuth fails"""
        mock_get_headers.side_effect = ChatGPTOAuthError(401, "No valid token")

        config = ChatGPTOAuthChatConfig()

        with pytest.raises(ChatGPTOAuthError) as exc_info:
            config.validate_environment(
                headers={},
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert exc_info.value.status_code == 401

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_api_base"
    )
    def test_get_complete_url_fallback_on_error(self, mock_get_base):
        """Test get_complete_url uses fallback when OAuth fails"""
        mock_get_base.side_effect = ChatGPTOAuthError(401, "No valid token")

        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gpt-4",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        # Should fall back to default OpenAI API
        assert url == "https://api.openai.com/v1/chat/completions"


class TestChatGPTOAuthChatChatGPTBackend:
    """Test ChatGPT backend mode specific functionality"""

    def setup_method(self):
        """Reset singleton before each test"""
        ChatGPTOAuthTokenManager._instance = None

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment_with_account_id(self, mock_get_headers):
        """Test validate_environment includes ChatGPT-Account-ID header in backend mode"""
        mock_get_headers.return_value = {
            "Authorization": "Bearer access-token",
            "Content-Type": "application/json",
            "ChatGPT-Account-ID": "account-123",
        }

        config = ChatGPTOAuthChatConfig()
        headers = config.validate_environment(
            headers={},
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )

        assert headers["Authorization"] == "Bearer access-token"
        assert headers["ChatGPT-Account-ID"] == "account-123"

    def test_get_complete_url_chatgpt_backend_base(self):
        """Test URL construction for ChatGPT backend"""
        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base="https://chatgpt.com/backend-api/codex",
            api_key=None,
            model="gpt-5.1-codex",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        # ChatGPT backend still uses /chat/completions
        assert url == "https://chatgpt.com/backend-api/codex/chat/completions"
        assert "/v1" not in url

    def test_get_complete_url_adds_v1_for_openai_api(self):
        """Test URL construction adds /v1 for OpenAI API when missing"""
        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base="https://api.openai.com",
            api_key=None,
            model="gpt-4",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://api.openai.com/v1/chat/completions"
