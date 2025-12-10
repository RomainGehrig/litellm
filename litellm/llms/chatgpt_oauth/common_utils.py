"""
Common utilities for ChatGPT OAuth provider.

This module handles OAuth token management for ChatGPT Plus authentication,
including reading tokens from the Codex CLI auth file and refreshing expired tokens.

Uses the ChatGPT backend API at chatgpt.com/backend-api/codex with OAuth access tokens.
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
DEFAULT_AUTH_BASE_URL = "https://auth.openai.com"

# ChatGPT backend API base URL
CHATGPT_BACKEND_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Token expiry buffer (refresh tokens 5 minutes before expiry)
TOKEN_EXPIRY_BUFFER_SECONDS = 300


class ChatGPTOAuthTokenManager:
    """
    Manages OAuth tokens for ChatGPT Plus authentication.

    This class handles:
    - Reading tokens from ~/.codex/auth.json or environment variables
    - Refreshing expired access tokens
    - Thread-safe token storage and updates
    - Providing authorization headers for the ChatGPT backend API
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
        self._access_token_expiry: Optional[float] = None

        # ChatGPT backend specific
        self._account_id: Optional[str] = None

        # Configuration
        self._client_id = get_secret_str("CHATGPT_OAUTH_CLIENT_ID") or DEFAULT_AUTH0_CLIENT_ID
        self._auth_base_url = get_secret_str("CHATGPT_OAUTH_AUTH_URL") or DEFAULT_AUTH_BASE_URL

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
        env_account_id = get_secret_str("CHATGPT_OAUTH_ACCOUNT_ID")

        if env_access_token or env_refresh_token:
            verbose_logger.debug("ChatGPT OAuth: Loading tokens from environment variables")
            self._access_token = env_access_token
            self._refresh_token = env_refresh_token
            self._account_id = env_account_id or self._extract_account_id_from_token()
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

            tokens = auth_data.get("tokens", auth_data)

            self._access_token = tokens.get("access_token")
            self._refresh_token = tokens.get("refresh_token")

            # Check for account ID (required for ChatGPT backend)
            # This can be extracted from the access token or stored separately
            self._account_id = (
                auth_data.get("account_id") or
                auth_data.get("chatgpt_account_id") or
                tokens.get("account_id") or
                self._extract_account_id_from_token()
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
                f"account_id={'present' if self._account_id else 'missing'}"
            )

        except Exception as e:
            verbose_logger.warning(f"ChatGPT OAuth: Failed to load auth file: {e}")

    def _extract_account_id_from_token(self) -> Optional[str]:
        """
        Extract account ID from the access token JWT.

        The access token from ChatGPT is a JWT that may contain the account ID
        in its claims (typically as 'https://api.openai.com/auth' claim with
        'user_id' or 'account_id' field).
        """
        if not self._access_token:
            return None

        try:
            import base64

            # JWT format: header.payload.signature
            parts = self._access_token.split(".")
            if len(parts) != 3:
                return None

            # Decode the payload (second part)
            # Add padding if needed
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            claims = json.loads(decoded)

            # Try various claim locations for account ID
            # OpenAI uses custom claims under 'https://api.openai.com/auth'
            auth_claim = claims.get("https://api.openai.com/auth", {})
            account_id = (
                auth_claim.get("account_id") or
                auth_claim.get("user_id") or
                claims.get("sub") or  # Standard JWT subject claim
                claims.get("account_id")
            )

            if account_id:
                verbose_logger.debug(f"ChatGPT OAuth: Extracted account_id from token")
                return account_id

        except Exception as e:
            verbose_logger.debug(f"ChatGPT OAuth: Failed to extract account_id from token: {e}")

        return None

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

            # Update refresh token if a new one is provided
            new_refresh_token = token_data.get("refresh_token")
            if new_refresh_token:
                self._refresh_token = new_refresh_token

            # Calculate expiry
            expires_in = token_data.get("expires_in", 3600)
            self._access_token_expiry = time.time() + expires_in

            # Extract account ID from new token if not already set
            if not self._account_id:
                self._account_id = self._extract_account_id_from_token()

            verbose_logger.debug("ChatGPT OAuth: Access token refreshed successfully")

            # Save updated tokens
            self._save_tokens_to_file()

        except httpx.HTTPError as e:
            raise ChatGPTOAuthError(
                status_code=500,
                message=f"Token refresh request failed: {str(e)}"
            )

    def get_access_token(self) -> str:
        """
        Get a valid access token for API requests.

        This method will refresh the token if it's expired or about to expire.

        Returns:
            str: The access token to use in the Authorization header

        Raises:
            ChatGPTOAuthError: If no valid token is available
        """
        with self._token_lock:
            # Check if access token needs refresh
            if self._is_token_expired() and self._refresh_token:
                self._refresh_access_token()

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
        """
        Get the API base URL for the ChatGPT backend.

        Returns:
            str: The API base URL (chatgpt.com/backend-api/codex)
        """
        # Check for explicit override first
        explicit_base = get_secret_str("CHATGPT_OAUTH_API_BASE")
        if explicit_base:
            return explicit_base

        return CHATGPT_BACKEND_BASE_URL

    def get_account_id(self) -> Optional[str]:
        """Get the ChatGPT account ID (required for ChatGPT backend)."""
        return self._account_id

    def get_authorization_headers(self) -> Dict[str, str]:
        """
        Get the authorization headers for ChatGPT backend API requests.

        Includes both Authorization and ChatGPT-Account-ID headers.

        Returns:
            Dict[str, str]: Headers to include in API requests

        Raises:
            ChatGPTOAuthError: If no valid token is available
        """
        token = self.get_access_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        if self._account_id:
            headers["ChatGPT-Account-ID"] = self._account_id
        else:
            verbose_logger.warning(
                "ChatGPT OAuth: No account_id available. "
                "Some ChatGPT backend endpoints may not work correctly."
            )

        return headers

    def reload_tokens(self) -> None:
        """Force reload tokens from the auth file."""
        with self._token_lock:
            self._load_tokens_from_file()

    def set_tokens(
        self,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> None:
        """Manually set tokens (useful for testing or programmatic configuration)."""
        with self._token_lock:
            if access_token:
                self._access_token = access_token
            if refresh_token:
                self._refresh_token = refresh_token
            if account_id:
                self._account_id = account_id


# Global token manager instance
_token_manager: Optional[ChatGPTOAuthTokenManager] = None


def get_token_manager() -> ChatGPTOAuthTokenManager:
    """Get the global token manager instance."""
    global _token_manager
    if _token_manager is None:
        _token_manager = ChatGPTOAuthTokenManager()
    return _token_manager


def get_chatgpt_oauth_headers() -> Dict[str, str]:
    """
    Get the complete authorization headers for ChatGPT OAuth.

    This includes the Authorization header and the ChatGPT-Account-ID header.

    Returns:
        Dict[str, str]: Headers to include in API requests

    Raises:
        ChatGPTOAuthError: If no valid token is available
    """
    manager = get_token_manager()
    return manager.get_authorization_headers()


def get_chatgpt_oauth_api_base() -> str:
    """
    Get the API base URL for ChatGPT OAuth.

    Returns:
        str: The API base URL (chatgpt.com/backend-api/codex)
    """
    manager = get_token_manager()
    return manager.get_api_base()
