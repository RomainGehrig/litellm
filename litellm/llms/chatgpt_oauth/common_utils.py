"""
Common utilities for ChatGPT OAuth provider.

This module handles OAuth token management for ChatGPT Plus authentication,
including reading tokens from the Codex CLI auth file, refreshing expired tokens,
and exchanging tokens for OpenAI API keys.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret_str


class ChatGPTOAuthError(Exception):
    """Exception raised for ChatGPT OAuth errors."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://auth.openai.com/oauth/token")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(self.message)


# Default OAuth configuration (matches Codex CLI)
DEFAULT_AUTH0_CLIENT_ID = "DRivsnm2Mu42T3KOpqdtwB3NYviHYzwD"
DEFAULT_AUTH0_AUDIENCE = "https://api.openai.com/v1"
DEFAULT_AUTH_BASE_URL = "https://auth.openai.com"

# Token expiry buffer (refresh tokens 5 minutes before expiry)
TOKEN_EXPIRY_BUFFER_SECONDS = 300


class ChatGPTOAuthTokenManager:
    """
    Manages OAuth tokens for ChatGPT Plus authentication.

    This class handles:
    - Reading tokens from ~/.codex/auth.json or environment variables
    - Refreshing expired access tokens
    - Exchanging id_token for OpenAI API keys
    - Thread-safe token storage and updates
    """

    _instance: Optional["ChatGPTOAuthTokenManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ChatGPTOAuthTokenManager":
        """Singleton pattern to ensure only one token manager exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the token manager."""
        if self._initialized:
            return

        self._initialized = True
        self._token_lock = threading.Lock()

        # Token storage
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._id_token: Optional[str] = None
        self._api_key: Optional[str] = None
        self._access_token_expiry: Optional[float] = None

        # Configuration
        self._client_id = get_secret_str("CHATGPT_OAUTH_CLIENT_ID") or DEFAULT_AUTH0_CLIENT_ID
        self._auth_base_url = get_secret_str("CHATGPT_OAUTH_AUTH_URL") or DEFAULT_AUTH_BASE_URL
        self._audience = get_secret_str("CHATGPT_OAUTH_AUDIENCE") or DEFAULT_AUTH0_AUDIENCE

        # Auth file path (default to ~/.codex/auth.json)
        self._auth_file_path = (
            get_secret_str("CHATGPT_OAUTH_AUTH_FILE") or
            str(Path.home() / ".codex" / "auth.json")
        )

        # Load tokens on initialization
        self._load_tokens_from_file()

    def _load_tokens_from_file(self) -> None:
        """Load tokens from the auth file (e.g., ~/.codex/auth.json)."""
        # First check environment variables
        env_access_token = get_secret_str("CHATGPT_OAUTH_ACCESS_TOKEN")
        env_refresh_token = get_secret_str("CHATGPT_OAUTH_REFRESH_TOKEN")
        env_id_token = get_secret_str("CHATGPT_OAUTH_ID_TOKEN")
        env_api_key = get_secret_str("CHATGPT_OAUTH_API_KEY")

        if env_access_token or env_refresh_token or env_api_key:
            verbose_logger.debug("ChatGPT OAuth: Loading tokens from environment variables")
            self._access_token = env_access_token
            self._refresh_token = env_refresh_token
            self._id_token = env_id_token
            self._api_key = env_api_key
            return

        # Try to load from auth file
        if not os.path.exists(self._auth_file_path):
            verbose_logger.debug(f"ChatGPT OAuth: Auth file not found at {self._auth_file_path}")
            return

        try:
            with open(self._auth_file_path, "r") as f:
                auth_data = json.load(f)

            verbose_logger.debug(f"ChatGPT OAuth: Loaded auth data from {self._auth_file_path}")

            # Extract tokens - the file structure may vary
            # Common structures:
            # 1. Direct keys: {"access_token": "...", "refresh_token": "...", ...}
            # 2. Nested: {"tokens": {"access_token": "...", ...}}
            # 3. With API key: {"OPENAI_API_KEY": "sk-...", ...}

            tokens = auth_data.get("tokens", auth_data)

            self._access_token = tokens.get("access_token")
            self._refresh_token = tokens.get("refresh_token")
            self._id_token = tokens.get("id_token")

            # Check for API key (may be stored after token exchange)
            self._api_key = (
                auth_data.get("OPENAI_API_KEY") or
                auth_data.get("api_key") or
                tokens.get("api_key")
            )

            # Parse expiry if present
            expires_at = tokens.get("expires_at")
            if expires_at:
                if isinstance(expires_at, str):
                    # ISO format timestamp
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                        self._access_token_expiry = dt.timestamp()
                    except Exception:
                        pass
                elif isinstance(expires_at, (int, float)):
                    self._access_token_expiry = float(expires_at)

            # If no expiry but we have expires_in, calculate it
            expires_in = tokens.get("expires_in")
            if expires_in and not self._access_token_expiry:
                self._access_token_expiry = time.time() + int(expires_in)

            verbose_logger.debug(
                f"ChatGPT OAuth: Tokens loaded - "
                f"access_token={'present' if self._access_token else 'missing'}, "
                f"refresh_token={'present' if self._refresh_token else 'missing'}, "
                f"api_key={'present' if self._api_key else 'missing'}"
            )

        except Exception as e:
            verbose_logger.warning(f"ChatGPT OAuth: Failed to load auth file: {e}")

    def _save_tokens_to_file(self) -> None:
        """Save updated tokens back to the auth file."""
        if not os.path.exists(self._auth_file_path):
            return

        try:
            with open(self._auth_file_path, "r") as f:
                auth_data = json.load(f)

            # Update tokens
            if "tokens" in auth_data:
                tokens = auth_data["tokens"]
            else:
                tokens = auth_data

            if self._access_token:
                tokens["access_token"] = self._access_token
            if self._refresh_token:
                tokens["refresh_token"] = self._refresh_token
            if self._id_token:
                tokens["id_token"] = self._id_token
            if self._api_key:
                auth_data["OPENAI_API_KEY"] = self._api_key
            if self._access_token_expiry:
                tokens["expires_at"] = int(self._access_token_expiry)

            # Write back
            with open(self._auth_file_path, "w") as f:
                json.dump(auth_data, f, indent=2)

            verbose_logger.debug(f"ChatGPT OAuth: Saved tokens to {self._auth_file_path}")

        except Exception as e:
            verbose_logger.warning(f"ChatGPT OAuth: Failed to save tokens: {e}")

    def _is_token_expired(self) -> bool:
        """Check if the access token is expired or about to expire."""
        if not self._access_token:
            return True
        if not self._access_token_expiry:
            # If we don't know the expiry, assume it might be expired
            return False
        return time.time() >= (self._access_token_expiry - TOKEN_EXPIRY_BUFFER_SECONDS)

    def _refresh_access_token(self) -> None:
        """Refresh the access token using the refresh token."""
        if not self._refresh_token:
            raise ChatGPTOAuthError(
                status_code=401,
                message="No refresh token available. Please re-authenticate with ChatGPT."
            )

        verbose_logger.debug("ChatGPT OAuth: Refreshing access token")

        token_url = f"{self._auth_base_url}/oauth/token"

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self._client_id,
        }

        try:
            response = httpx.post(
                token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )

            if response.status_code != 200:
                verbose_logger.error(
                    f"ChatGPT OAuth: Token refresh failed - {response.status_code}: {response.text}"
                )
                raise ChatGPTOAuthError(
                    status_code=response.status_code,
                    message=f"Token refresh failed: {response.text}"
                )

            token_data = response.json()

            self._access_token = token_data.get("access_token")
            self._id_token = token_data.get("id_token")

            # Update refresh token if a new one is provided
            new_refresh_token = token_data.get("refresh_token")
            if new_refresh_token:
                self._refresh_token = new_refresh_token

            # Calculate expiry
            expires_in = token_data.get("expires_in", 3600)
            self._access_token_expiry = time.time() + expires_in

            verbose_logger.debug("ChatGPT OAuth: Access token refreshed successfully")

            # Save updated tokens
            self._save_tokens_to_file()

            # If we have a new id_token, we may need to exchange it for a new API key
            if self._id_token and not self._api_key:
                self._exchange_token_for_api_key()

        except httpx.HTTPError as e:
            raise ChatGPTOAuthError(
                status_code=500,
                message=f"Token refresh request failed: {str(e)}"
            )

    def _exchange_token_for_api_key(self) -> None:
        """
        Exchange the id_token for an OpenAI API key using token exchange grant.

        This follows the OAuth 2.0 Token Exchange specification (RFC 8693).
        The exchanged API key can be used with standard OpenAI API endpoints.
        """
        if not self._id_token:
            verbose_logger.debug("ChatGPT OAuth: No id_token available for exchange")
            return

        verbose_logger.debug("ChatGPT OAuth: Exchanging id_token for API key")

        token_url = f"{self._auth_base_url}/oauth/token"

        payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": self._id_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
            "requested_token_type": "openai-api-key",
            "client_id": self._client_id,
            "audience": self._audience,
        }

        try:
            response = httpx.post(
                token_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )

            if response.status_code != 200:
                verbose_logger.warning(
                    f"ChatGPT OAuth: Token exchange failed - {response.status_code}: {response.text}"
                )
                # Don't raise - we can still try using access_token directly
                return

            exchange_data = response.json()

            # The API key is returned in the access_token field
            self._api_key = exchange_data.get("access_token")

            if self._api_key:
                verbose_logger.debug("ChatGPT OAuth: API key obtained via token exchange")
                self._save_tokens_to_file()

        except httpx.HTTPError as e:
            verbose_logger.warning(f"ChatGPT OAuth: Token exchange request failed: {e}")

    def get_authorization_token(self) -> str:
        """
        Get a valid authorization token for API requests.

        This method will:
        1. Return the API key if available (preferred)
        2. Otherwise return the access token, refreshing if needed

        Returns:
            str: The token to use in the Authorization header

        Raises:
            ChatGPTOAuthError: If no valid token is available
        """
        with self._token_lock:
            # Prefer API key if available
            if self._api_key:
                return self._api_key

            # Check if access token needs refresh
            if self._is_token_expired() and self._refresh_token:
                self._refresh_access_token()

            # Try token exchange if we have id_token but no API key
            if self._id_token and not self._api_key:
                self._exchange_token_for_api_key()
                if self._api_key:
                    return self._api_key

            # Fall back to access token
            if self._access_token:
                return self._access_token

            raise ChatGPTOAuthError(
                status_code=401,
                message=(
                    "No valid ChatGPT OAuth token available. "
                    "Please authenticate using 'codex login' or set CHATGPT_OAUTH_* environment variables."
                )
            )

    def get_api_base(self) -> str:
        """Get the API base URL."""
        return get_secret_str("CHATGPT_OAUTH_API_BASE") or "https://api.openai.com/v1"

    def reload_tokens(self) -> None:
        """Force reload tokens from the auth file."""
        with self._token_lock:
            self._load_tokens_from_file()

    def set_tokens(
        self,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        id_token: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Manually set tokens (useful for testing or programmatic configuration)."""
        with self._token_lock:
            if access_token:
                self._access_token = access_token
            if refresh_token:
                self._refresh_token = refresh_token
            if id_token:
                self._id_token = id_token
            if api_key:
                self._api_key = api_key


# Global token manager instance
_token_manager: Optional[ChatGPTOAuthTokenManager] = None


def get_token_manager() -> ChatGPTOAuthTokenManager:
    """Get the global token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = ChatGPTOAuthTokenManager()
    return _token_manager


def get_chatgpt_oauth_credentials() -> Tuple[str, str]:
    """
    Get the API base and authorization token for ChatGPT OAuth.

    Returns:
        Tuple[str, str]: (api_base, authorization_token)
    """
    manager = get_token_manager()
    return manager.get_api_base(), manager.get_authorization_token()
