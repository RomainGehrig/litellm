"""
ChatGPT OAuth Chat Completion transformation.

This module provides the configuration class for making chat completion requests
using ChatGPT Plus OAuth credentials.

Uses the ChatGPT backend API at chatgpt.com/backend-api/codex with OAuth access tokens.

Note: The ChatGPT backend typically uses the Responses API (/responses) rather than
Chat Completions (/chat/completions) for Codex models. This chat config is mainly
for compatibility with models that still use the chat completions endpoint.
"""

from typing import List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues

from ..common_utils import (
    ChatGPTOAuthError,
    CHATGPT_BACKEND_BASE_URL,
    get_chatgpt_oauth_api_base,
    get_chatgpt_oauth_headers,
    get_token_manager,
)


class ChatGPTOAuthChatConfig(OpenAIGPTConfig):
    """
    Configuration for ChatGPT OAuth Chat API.

    This class extends OpenAIGPTConfig to use OAuth tokens from ChatGPT Plus
    instead of a static API key. It handles:
    - Reading OAuth tokens from ~/.codex/auth.json or environment variables
    - Automatic token refresh when tokens expire
    - Providing ChatGPT-Account-ID header for the ChatGPT backend

    The class is designed to be a drop-in replacement for OpenAI chat completions
    but using ChatGPT Plus subscription credentials.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._custom_llm_provider = "chatgpt_oauth"

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return self._custom_llm_provider

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API base and key for ChatGPT OAuth provider.

        This method overrides the parent to use OAuth token management
        instead of static API keys.

        Args:
            api_base: Optional API base URL override
            api_key: Optional API key override (not used for OAuth)

        Returns:
            Tuple of (api_base, api_key) where api_key is the OAuth token
        """
        try:
            manager = get_token_manager()
            oauth_api_base = manager.get_api_base()
            oauth_token = manager.get_access_token()

            # Use provided api_base if specified, otherwise use OAuth api_base
            resolved_api_base = api_base or oauth_api_base

            verbose_logger.debug(
                f"ChatGPT OAuth Chat: api_base={resolved_api_base}, "
                f"token={'present' if oauth_token else 'missing'}"
            )

            return resolved_api_base, oauth_token

        except ChatGPTOAuthError as e:
            verbose_logger.error(f"ChatGPT OAuth: Failed to get credentials - {e.message}")
            raise

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate and setup environment for ChatGPT OAuth requests.

        This method gets the OAuth credentials and sets up the appropriate headers:
        - Authorization header with the OAuth access token
        - ChatGPT-Account-ID header for the ChatGPT backend

        Args:
            headers: Request headers dict to update
            model: Model name
            messages: Chat messages
            optional_params: Optional parameters
            litellm_params: LiteLLM specific parameters
            api_key: Optional API key (ignored, OAuth token is used)
            api_base: Optional API base URL

        Returns:
            Updated headers dict with Authorization and ChatGPT-Account-ID headers
        """
        try:
            oauth_headers = get_chatgpt_oauth_headers()
            headers.update(oauth_headers)

            return headers

        except ChatGPTOAuthError as e:
            verbose_logger.error(f"ChatGPT OAuth: validate_environment failed - {e.message}")
            raise

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the ChatGPT OAuth API call.

        The ChatGPT backend typically uses the Responses API (/responses)
        rather than Chat Completions for Codex models.

        Args:
            api_base: API base URL
            api_key: Not used (OAuth handles auth)
            model: Model name
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            stream: Whether streaming is enabled

        Returns:
            Complete URL for the API call
        """
        if api_base is None:
            try:
                api_base = get_chatgpt_oauth_api_base()
            except ChatGPTOAuthError:
                # Fall back to default ChatGPT backend
                api_base = CHATGPT_BACKEND_BASE_URL

        # Remove trailing slash from api_base if present
        api_base = api_base.rstrip("/")

        # ChatGPT backend doesn't typically support /chat/completions
        # For Codex models, it uses /responses instead
        # But we still construct the URL in case it's needed
        verbose_logger.warning(
            "ChatGPT OAuth Chat: ChatGPT backend typically uses /responses, not /chat/completions"
        )
        return f"{api_base}/chat/completions"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Get the OAuth token for API requests.

        This overrides the parent method to use OAuth token management
        instead of looking for static API keys.

        Args:
            api_key: Ignored parameter (for interface compatibility)

        Returns:
            OAuth token from token manager
        """
        try:
            manager = get_token_manager()
            return manager.get_access_token()
        except ChatGPTOAuthError:
            return None

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        """
        Get the API base URL.

        Args:
            api_base: Optional override for API base

        Returns:
            API base URL (override or default from OAuth config)
        """
        if api_base:
            return api_base
        try:
            return get_chatgpt_oauth_api_base()
        except ChatGPTOAuthError:
            return CHATGPT_BACKEND_BASE_URL
