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
    ChatGPTOAuthError,
    ChatGPTOAuthTokenManager,
    CHATGPT_BACKEND_BASE_URL,
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
        "litellm.llms.chatgpt_oauth.chat.transformation.get_token_manager"
    )
    def test_get_openai_compatible_provider_info(self, mock_get_manager):
        """Test _get_openai_compatible_provider_info returns OAuth credentials"""
        mock_manager = MagicMock()
        mock_manager.get_api_base.return_value = CHATGPT_BACKEND_BASE_URL
        mock_manager.get_access_token.return_value = "oauth-access-token"
        mock_get_manager.return_value = mock_manager

        config = ChatGPTOAuthChatConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )

        assert api_base == CHATGPT_BACKEND_BASE_URL
        assert api_key == "oauth-access-token"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_token_manager"
    )
    def test_get_openai_compatible_provider_info_with_override(self, mock_get_manager):
        """Test _get_openai_compatible_provider_info respects api_base override"""
        mock_manager = MagicMock()
        mock_manager.get_api_base.return_value = CHATGPT_BACKEND_BASE_URL
        mock_manager.get_access_token.return_value = "oauth-access-token"
        mock_get_manager.return_value = mock_manager

        config = ChatGPTOAuthChatConfig()
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base="https://custom.api.com/codex", api_key=None
        )

        # Should use provided api_base but OAuth token
        assert api_base == "https://custom.api.com/codex"
        assert api_key == "oauth-access-token"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment(self, mock_get_headers):
        """Test validate_environment sets correct headers"""
        mock_get_headers.return_value = {
            "Authorization": "Bearer oauth-access-token",
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

        assert headers["Authorization"] == "Bearer oauth-access-token"
        assert headers["Content-Type"] == "application/json"
        assert headers["ChatGPT-Account-ID"] == "account-123"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment_preserves_existing_headers(self, mock_get_headers):
        """Test validate_environment preserves existing headers"""
        mock_get_headers.return_value = {
            "Authorization": "Bearer oauth-access-token",
            "Content-Type": "application/json",
            "ChatGPT-Account-ID": "account-123",
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
        assert headers["Authorization"] == "Bearer oauth-access-token"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_api_base"
    )
    def test_get_complete_url_default(self, mock_get_base):
        """Test get_complete_url returns correct endpoint"""
        mock_get_base.return_value = CHATGPT_BACKEND_BASE_URL

        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gpt-4",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == f"{CHATGPT_BACKEND_BASE_URL}/chat/completions"

    def test_get_complete_url_with_custom_base(self):
        """Test get_complete_url with custom api_base"""
        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base="https://custom.api.com/codex",
            api_key=None,
            model="gpt-4",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://custom.api.com/codex/chat/completions"

    def test_get_complete_url_handles_trailing_slash(self):
        """Test get_complete_url handles trailing slash in api_base"""
        config = ChatGPTOAuthChatConfig()
        url = config.get_complete_url(
            api_base=f"{CHATGPT_BACKEND_BASE_URL}/",
            api_key=None,
            model="gpt-4",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == f"{CHATGPT_BACKEND_BASE_URL}/chat/completions"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_token_manager"
    )
    def test_get_api_key_returns_oauth_token(self, mock_get_manager):
        """Test get_api_key returns OAuth token"""
        mock_manager = MagicMock()
        mock_manager.get_access_token.return_value = "oauth-access-token"
        mock_get_manager.return_value = mock_manager

        token = ChatGPTOAuthChatConfig.get_api_key()

        assert token == "oauth-access-token"

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_token_manager"
    )
    def test_get_api_key_returns_none_on_error(self, mock_get_manager):
        """Test get_api_key returns None when OAuth fails"""
        mock_manager = MagicMock()
        mock_manager.get_access_token.side_effect = ChatGPTOAuthError(401, "No token")
        mock_get_manager.return_value = mock_manager

        token = ChatGPTOAuthChatConfig.get_api_key()

        assert token is None

    @patch(
        "litellm.llms.chatgpt_oauth.chat.transformation.get_chatgpt_oauth_api_base"
    )
    def test_get_api_base_default(self, mock_get_base):
        """Test get_api_base returns OAuth api_base"""
        mock_get_base.return_value = CHATGPT_BACKEND_BASE_URL

        api_base = ChatGPTOAuthChatConfig.get_api_base()

        assert api_base == CHATGPT_BACKEND_BASE_URL

    def test_get_api_base_with_override(self):
        """Test get_api_base respects override"""
        api_base = ChatGPTOAuthChatConfig.get_api_base(
            api_base="https://custom.api.com/codex"
        )

        assert api_base == "https://custom.api.com/codex"

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

        # Should fall back to default ChatGPT backend
        assert url == f"{CHATGPT_BACKEND_BASE_URL}/chat/completions"
