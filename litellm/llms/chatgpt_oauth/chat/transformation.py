"""
ChatGPT OAuth Chat Completion transformation.

This module provides the configuration class for making chat completion requests
using ChatGPT Plus OAuth credentials.

Supports two modes:
1. ChatGPT Backend mode: Uses OAuth access token directly with chatgpt.com/backend-api/codex
2. Standard OpenAI mode: Uses exchanged API key with api.openai.com/v1

Note: The ChatGPT backend typically uses the Responses API (/responses) rather than
Chat Completions (/chat/completions) for Codex models. This chat config is mainly
for compatibility with models that still use the chat completions endpoint.
"""

from typing import List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues

from ..common_utils import (
    ChatGPTBackendMode,
    ChatGPTOAuthError,
    CHATGPT_BACKEND_BASE_URL,
    OPENAI_API_BASE_URL,
    get_chatgpt_oauth_api_base,
    get_chatgpt_oauth_credentials,
    get_chatgpt_oauth_headers,
    get_chatgpt_oauth_mode,
)


class ChatGPTOAuthChatConfig(OpenAIGPTConfig):
    """
    Configuration for ChatGPT OAuth Chat API.

    This class extends OpenAIGPTConfig to use OAuth tokens from ChatGPT Plus
    instead of a static API key. It handles:
    - Reading OAuth tokens from ~/.codex/auth.json or environment variables
    - Automatic token refresh when tokens expire
    - Token exchange to get API keys when needed
    - Support for both ChatGPT backend and standard OpenAI API modes

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
            oauth_api_base, oauth_token = get_chatgpt_oauth_credentials()

            # Use provided api_base if specified, otherwise use OAuth api_base
            resolved_api_base = api_base or oauth_api_base

            mode = get_chatgpt_oauth_mode()
            verbose_logger.debug(
                f"ChatGPT OAuth Chat: Using mode={mode.value}, api_base={resolved_api_base}, "
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

        This method gets the OAuth credentials and sets up the appropriate headers
        based on the backend mode:
        - For ChatGPT backend: Authorization + ChatGPT-Account-ID headers
        - For OpenAI API: Authorization header only

        Args:
            headers: Request headers dict to update
            model: Model name
            messages: Chat messages
            optional_params: Optional parameters
            litellm_params: LiteLLM specific parameters
            api_key: Optional API key (ignored, OAuth token is used)
            api_base: Optional API base URL

        Returns:
            Updated headers dict with Authorization (and optionally ChatGPT-Account-ID) header
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

        The URL structure differs based on the backend mode:
        - ChatGPT backend: Not typically used for chat/completions (uses Responses API)
        - OpenAI API: api.openai.com/v1/chat/completions

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
                # Fall back to default OpenAI API
                api_base = OPENAI_API_BASE_URL

        # Remove trailing slash from api_base if present
        api_base = api_base.rstrip("/")

        # Determine if we're using ChatGPT backend or OpenAI API
        is_chatgpt_backend = "backend-api" in api_base or api_base == CHATGPT_BACKEND_BASE_URL

        if is_chatgpt_backend:
            # ChatGPT backend doesn't typically support /chat/completions
            # For Codex models, it uses /responses instead
            # But we still construct the URL in case it's needed
            verbose_logger.warning(
                "ChatGPT OAuth Chat: ChatGPT backend typically uses /responses, not /chat/completions"
            )
            return f"{api_base}/chat/completions"
        else:
            # Standard OpenAI API: api.openai.com/v1/chat/completions
            endpoint = "chat/completions"

            # Check if endpoint is already in the api_base
            if endpoint in api_base:
                return api_base

            # Ensure we have /v1 in the path for OpenAI
            if "api.openai.com" in api_base and "/v1" not in api_base:
                api_base = f"{api_base}/v1"

            return f"{api_base}/{endpoint}"

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
            _, token = get_chatgpt_oauth_credentials()
            return token
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
            return OPENAI_API_BASE_URL
