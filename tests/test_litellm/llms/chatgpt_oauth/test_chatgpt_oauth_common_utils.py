"""
Tests for ChatGPT OAuth common utilities.

Tests the OAuth token management functionality including:
- Token loading from auth files
- Token refresh
- Token exchange for API keys
- Singleton pattern for token manager

Source: litellm/llms/chatgpt_oauth/common_utils.py
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

from litellm.llms.chatgpt_oauth.common_utils import (
    ChatGPTOAuthError,
    ChatGPTOAuthTokenManager,
    get_chatgpt_oauth_credentials,
    get_token_manager,
)


class TestChatGPTOAuthTokenManager:
    """Test ChatGPT OAuth token manager functionality"""

    def test_token_manager_singleton_pattern(self):
        """Test that token manager uses singleton pattern"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager1 = get_token_manager()
        manager2 = get_token_manager()

        assert manager1 is manager2, "Token manager should be a singleton"

    def test_token_manager_initialization(self):
        """Test token manager initializes with expected attributes"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()

        assert hasattr(manager, "_access_token")
        assert hasattr(manager, "_refresh_token")
        assert hasattr(manager, "_id_token")
        assert hasattr(manager, "_api_key")
        assert hasattr(manager, "_auth_file_path")
        assert hasattr(manager, "_client_id")

    @patch.dict(
        os.environ,
        {
            "CHATGPT_OAUTH_ACCESS_TOKEN": "test-access-token",
            "CHATGPT_OAUTH_REFRESH_TOKEN": "test-refresh-token",
            "CHATGPT_OAUTH_API_KEY": "sk-test-api-key",
        },
        clear=False,
    )
    def test_load_tokens_from_environment(self):
        """Test loading tokens from environment variables"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()

        # Environment variables should be loaded
        assert manager._access_token == "test-access-token"
        assert manager._refresh_token == "test-refresh-token"
        assert manager._api_key == "sk-test-api-key"

    def test_load_tokens_from_auth_file(self):
        """Test loading tokens from auth file"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        # Create temporary auth file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            auth_data = {
                "access_token": "file-access-token",
                "refresh_token": "file-refresh-token",
                "id_token": "file-id-token",
                "OPENAI_API_KEY": "sk-file-api-key",
            }
            json.dump(auth_data, f)
            temp_path = f.name

        try:
            # Clear env vars that might interfere and set auth file path via env
            with patch.dict(
                os.environ,
                {
                    "CHATGPT_OAUTH_ACCESS_TOKEN": "",
                    "CHATGPT_OAUTH_REFRESH_TOKEN": "",
                    "CHATGPT_OAUTH_API_KEY": "",
                    "CHATGPT_OAUTH_AUTH_FILE": temp_path,
                },
                clear=False,
            ):
                manager = ChatGPTOAuthTokenManager()
                # Manually set the auth file path and reload
                manager._auth_file_path = temp_path
                manager._load_tokens_from_file()

                assert manager._access_token == "file-access-token"
                assert manager._refresh_token == "file-refresh-token"
                assert manager._id_token == "file-id-token"
                assert manager._api_key == "sk-file-api-key"
        finally:
            os.unlink(temp_path)

    def test_set_tokens_manually(self):
        """Test manually setting tokens"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()

        manager.set_tokens(
            access_token="manual-access",
            refresh_token="manual-refresh",
            id_token="manual-id",
            api_key="sk-manual-key",
        )

        assert manager._access_token == "manual-access"
        assert manager._refresh_token == "manual-refresh"
        assert manager._id_token == "manual-id"
        assert manager._api_key == "sk-manual-key"

    def test_get_authorization_token_prefers_api_key(self):
        """Test that get_authorization_token prefers API key over access token"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()
        manager._api_key = "sk-preferred-key"
        manager._access_token = "access-token"

        token = manager.get_authorization_token()

        assert token == "sk-preferred-key", "Should prefer API key over access token"

    def test_get_authorization_token_falls_back_to_access_token(self):
        """Test that get_authorization_token falls back to access token"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()
        manager._api_key = None
        manager._access_token = "access-token"
        manager._access_token_expiry = None  # No expiry known

        token = manager.get_authorization_token()

        assert token == "access-token", "Should fall back to access token"

    def test_get_authorization_token_raises_without_tokens(self):
        """Test that get_authorization_token raises error when no tokens available"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()
        manager._api_key = None
        manager._access_token = None
        manager._refresh_token = None

        with pytest.raises(ChatGPTOAuthError) as exc_info:
            manager.get_authorization_token()

        assert exc_info.value.status_code == 401
        assert "No valid ChatGPT OAuth token" in exc_info.value.message

    @patch("litellm.llms.chatgpt_oauth.common_utils.httpx.post")
    def test_refresh_access_token(self, mock_post):
        """Test refreshing access token"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        # Mock successful refresh response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "id_token": "new-id-token",
            "expires_in": 3600,
        }
        mock_post.return_value = mock_response

        manager = ChatGPTOAuthTokenManager()
        manager._refresh_token = "old-refresh-token"
        manager._access_token_expiry = 0  # Expired

        manager._refresh_access_token()

        assert manager._access_token == "new-access-token"
        assert manager._refresh_token == "new-refresh-token"
        assert manager._id_token == "new-id-token"

    def test_refresh_without_refresh_token_raises(self):
        """Test that refresh raises error when no refresh token available"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()
        manager._refresh_token = None

        with pytest.raises(ChatGPTOAuthError) as exc_info:
            manager._refresh_access_token()

        assert exc_info.value.status_code == 401
        assert "No refresh token available" in exc_info.value.message

    def test_get_api_base_returns_default(self):
        """Test that get_api_base returns default OpenAI API base"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()
        api_base = manager.get_api_base()

        assert api_base == "https://api.openai.com/v1"

    @patch.dict(
        os.environ, {"CHATGPT_OAUTH_API_BASE": "https://custom.api.com/v1"}, clear=False
    )
    def test_get_api_base_from_environment(self):
        """Test that get_api_base reads from environment"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = ChatGPTOAuthTokenManager()
        api_base = manager.get_api_base()

        assert api_base == "https://custom.api.com/v1"


class TestChatGPTOAuthCredentials:
    """Test get_chatgpt_oauth_credentials helper function"""

    def test_get_credentials_returns_tuple(self):
        """Test that get_chatgpt_oauth_credentials returns api_base and token"""
        # Reset singleton for clean test
        ChatGPTOAuthTokenManager._instance = None

        manager = get_token_manager()
        manager._api_key = "sk-test-key"

        api_base, token = get_chatgpt_oauth_credentials()

        assert isinstance(api_base, str)
        assert isinstance(token, str)
        assert token == "sk-test-key"


class TestChatGPTOAuthError:
    """Test ChatGPT OAuth error class"""

    def test_error_has_status_code(self):
        """Test that error has status code attribute"""
        error = ChatGPTOAuthError(status_code=401, message="Unauthorized")

        assert error.status_code == 401
        assert error.message == "Unauthorized"

    def test_error_has_httpx_request_response(self):
        """Test that error has httpx request and response objects"""
        error = ChatGPTOAuthError(status_code=500, message="Server error")

        assert error.request is not None
        assert error.response is not None
        assert error.response.status_code == 500
