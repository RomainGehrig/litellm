"""
Tests for ChatGPT OAuth Responses API transformation.

Tests the ChatGPTOAuthResponsesAPIConfig class that handles ChatGPT OAuth-specific
transformations for the Responses API (used by GPT-5/Codex models).

Source: litellm/llms/chatgpt_oauth/responses/transformation.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.chatgpt_oauth.responses.transformation import (
    ChatGPTOAuthResponsesAPIConfig,
)
from litellm.llms.chatgpt_oauth.common_utils import (
    ChatGPTOAuthError,
    ChatGPTOAuthTokenManager,
    CHATGPT_BACKEND_BASE_URL,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


class TestChatGPTOAuthResponsesAPITransformation:
    """Test ChatGPT OAuth Responses API configuration and transformations"""

    def setup_method(self):
        """Reset singleton before each test"""
        ChatGPTOAuthTokenManager._instance = None

    def test_provider_config_registration(self):
        """Test that ChatGPT OAuth provider returns ChatGPTOAuthResponsesAPIConfig"""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="gpt-5.1-codex",
            provider=LlmProviders.CHATGPT_OAUTH,
        )

        assert (
            config is not None
        ), "Config should not be None for ChatGPT OAuth provider"
        assert isinstance(
            config, ChatGPTOAuthResponsesAPIConfig
        ), f"Expected ChatGPTOAuthResponsesAPIConfig, got {type(config)}"

    def test_custom_llm_provider_property(self):
        """Test that custom_llm_provider returns correct LlmProviders enum"""
        config = ChatGPTOAuthResponsesAPIConfig()

        assert config.custom_llm_provider == LlmProviders.CHATGPT_OAUTH

    @patch(
        "litellm.llms.chatgpt_oauth.responses.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment(self, mock_get_headers):
        """Test validate_environment sets correct headers"""
        mock_get_headers.return_value = {
            "Authorization": "Bearer oauth-access-token",
            "Content-Type": "application/json",
            "ChatGPT-Account-ID": "account-123",
        }

        config = ChatGPTOAuthResponsesAPIConfig()
        headers = config.validate_environment(
            headers={},
            model="gpt-5.1-codex",
            litellm_params=None,
        )

        assert headers["Authorization"] == "Bearer oauth-access-token"
        assert headers["Content-Type"] == "application/json"
        assert headers["ChatGPT-Account-ID"] == "account-123"

    @patch(
        "litellm.llms.chatgpt_oauth.responses.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment_preserves_existing_headers(self, mock_get_headers):
        """Test validate_environment preserves existing headers"""
        mock_get_headers.return_value = {
            "Authorization": "Bearer oauth-access-token",
            "Content-Type": "application/json",
            "ChatGPT-Account-ID": "account-123",
        }

        config = ChatGPTOAuthResponsesAPIConfig()
        headers = config.validate_environment(
            headers={"X-Custom-Header": "custom-value"},
            model="gpt-5.1-codex",
            litellm_params=None,
        )

        assert headers["X-Custom-Header"] == "custom-value"
        assert headers["Authorization"] == "Bearer oauth-access-token"

    @patch(
        "litellm.llms.chatgpt_oauth.responses.transformation.get_chatgpt_oauth_api_base"
    )
    def test_get_complete_url_default(self, mock_get_base):
        """Test get_complete_url returns correct responses endpoint"""
        mock_get_base.return_value = CHATGPT_BACKEND_BASE_URL

        config = ChatGPTOAuthResponsesAPIConfig()
        url = config.get_complete_url(api_base=None, litellm_params={})

        assert url == f"{CHATGPT_BACKEND_BASE_URL}/responses"

    def test_get_complete_url_with_custom_base(self):
        """Test get_complete_url with custom api_base"""
        config = ChatGPTOAuthResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://custom.api.com/codex", litellm_params={}
        )

        assert url == "https://custom.api.com/codex/responses"

    def test_get_complete_url_handles_trailing_slash(self):
        """Test get_complete_url handles trailing slash in api_base"""
        config = ChatGPTOAuthResponsesAPIConfig()
        url = config.get_complete_url(
            api_base=f"{CHATGPT_BACKEND_BASE_URL}/", litellm_params={}
        )

        assert url == f"{CHATGPT_BACKEND_BASE_URL}/responses"

    def test_get_complete_url_chatgpt_backend(self):
        """Test get_complete_url for ChatGPT backend"""
        config = ChatGPTOAuthResponsesAPIConfig()
        url = config.get_complete_url(
            api_base=CHATGPT_BACKEND_BASE_URL, litellm_params={}
        )

        assert url == f"{CHATGPT_BACKEND_BASE_URL}/responses"

    def test_inherits_from_openai_responses_config(self):
        """Test that ChatGPTOAuthResponsesAPIConfig inherits OpenAI config"""
        from litellm.llms.openai.responses.transformation import (
            OpenAIResponsesAPIConfig,
        )

        config = ChatGPTOAuthResponsesAPIConfig()

        assert isinstance(config, OpenAIResponsesAPIConfig)
        assert hasattr(config, "get_supported_openai_params")
        assert hasattr(config, "map_openai_params")
        assert hasattr(config, "transform_responses_api_request")
        assert hasattr(config, "transform_response_api_response")

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns expected parameters"""
        config = ChatGPTOAuthResponsesAPIConfig()

        supported = config.get_supported_openai_params("gpt-5.1-codex")

        # Should include standard OpenAI Responses API parameters
        expected_params = [
            "model",
            "input",
            "instructions",
            "temperature",
            "max_output_tokens",
            "stream",
            "tools",
            "tool_choice",
        ]

        for param in expected_params:
            assert param in supported, f"{param} should be in supported params"

    def test_map_openai_params_passthrough(self):
        """Test that map_openai_params passes through parameters"""
        config = ChatGPTOAuthResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            temperature=0.7,
            max_output_tokens=1000,
            stream=False,
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="gpt-5.1-codex",
            drop_params=False,
        )

        assert result.get("temperature") == 0.7
        assert result.get("max_output_tokens") == 1000
        assert result.get("stream") is False

    def test_get_delete_response_url(self):
        """Test get_delete_response_url constructs correct URL"""
        config = ChatGPTOAuthResponsesAPIConfig()
        url = config.get_delete_response_url(
            api_base=CHATGPT_BACKEND_BASE_URL,
            response_id="resp_abc123",
            litellm_params={},
        )

        assert url == f"{CHATGPT_BACKEND_BASE_URL}/responses/resp_abc123"

    def test_get_get_response_url(self):
        """Test get_get_response_url constructs correct URL"""
        config = ChatGPTOAuthResponsesAPIConfig()
        url = config.get_get_response_url(
            api_base=CHATGPT_BACKEND_BASE_URL,
            response_id="resp_abc123",
            litellm_params={},
        )

        assert url == f"{CHATGPT_BACKEND_BASE_URL}/responses/resp_abc123"


class TestChatGPTOAuthResponsesAPIValidation:
    """Test validation and error handling for Responses API"""

    def setup_method(self):
        """Reset singleton before each test"""
        ChatGPTOAuthTokenManager._instance = None

    @patch(
        "litellm.llms.chatgpt_oauth.responses.transformation.get_chatgpt_oauth_headers"
    )
    def test_validate_environment_raises_on_oauth_error(self, mock_get_headers):
        """Test validate_environment raises when OAuth fails"""
        mock_get_headers.side_effect = ChatGPTOAuthError(401, "No valid token")

        config = ChatGPTOAuthResponsesAPIConfig()

        with pytest.raises(ChatGPTOAuthError) as exc_info:
            config.validate_environment(
                headers={},
                model="gpt-5.1-codex",
                litellm_params=None,
            )

        assert exc_info.value.status_code == 401

    @patch(
        "litellm.llms.chatgpt_oauth.responses.transformation.get_chatgpt_oauth_api_base"
    )
    def test_get_complete_url_fallback_on_error(self, mock_get_base):
        """Test get_complete_url uses fallback when OAuth fails"""
        mock_get_base.side_effect = ChatGPTOAuthError(401, "No valid token")

        config = ChatGPTOAuthResponsesAPIConfig()
        url = config.get_complete_url(api_base=None, litellm_params={})

        # Should fall back to default ChatGPT backend
        assert url == f"{CHATGPT_BACKEND_BASE_URL}/responses"
