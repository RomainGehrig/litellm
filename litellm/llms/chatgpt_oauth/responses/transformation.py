"""
ChatGPT OAuth Responses API transformation.

This module provides the configuration class for making Responses API requests
(OpenAI's newer API for GPT-5/Codex models) using ChatGPT Plus OAuth credentials.

Supports two modes:
1. ChatGPT Backend mode: Uses OAuth access token directly with chatgpt.com/backend-api/codex/responses
2. Standard OpenAI mode: Uses exchanged API key with api.openai.com/v1/responses
"""

from typing import Optional

from litellm._logging import verbose_logger
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import (
    ChatGPTBackendMode,
    ChatGPTOAuthError,
    CHATGPT_BACKEND_BASE_URL,
    OPENAI_API_BASE_URL,
    get_chatgpt_oauth_api_base,
    get_chatgpt_oauth_headers,
    get_chatgpt_oauth_mode,
    get_token_manager,
)


class ChatGPTOAuthResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for ChatGPT OAuth Responses API.

    This class extends OpenAIResponsesAPIConfig to use OAuth tokens from ChatGPT Plus
    instead of a static API key. It's designed for GPT-5/Codex models that use the
    newer Responses API endpoint (/responses).

    Supports two backend modes:
    - ChatGPT Backend: Uses chatgpt.com/backend-api/codex/responses with OAuth tokens
    - OpenAI API: Uses api.openai.com/v1/responses with exchanged API key

    The class handles:
    - Reading OAuth tokens from ~/.codex/auth.json or environment variables
    - Automatic token refresh when tokens expire
    - Token exchange to get API keys when needed
    - Mode auto-detection based on available credentials
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.CHATGPT_OAUTH

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        """
        Validate and setup environment for ChatGPT OAuth Responses API requests.

        This method gets the OAuth credentials and sets up the appropriate headers
        based on the backend mode:
        - For ChatGPT backend: Authorization + ChatGPT-Account-ID headers
        - For OpenAI API: Authorization header only

        Args:
            headers: Request headers dict to update
            model: Model name
            litellm_params: LiteLLM specific parameters

        Returns:
            Updated headers dict with Authorization (and optionally ChatGPT-Account-ID) header
        """
        try:
            oauth_headers = get_chatgpt_oauth_headers()
            headers.update(oauth_headers)

            mode = get_chatgpt_oauth_mode()
            verbose_logger.debug(f"ChatGPT OAuth Responses: Using mode {mode.value}")

            return headers

        except ChatGPTOAuthError as e:
            verbose_logger.error(f"ChatGPT OAuth Responses: validate_environment failed - {e.message}")
            raise

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for the ChatGPT OAuth Responses API call.

        The URL structure differs based on the backend mode:
        - ChatGPT backend: {base}/responses (base already includes /backend-api/codex)
        - OpenAI API: {base}/responses (base includes /v1)

        Args:
            api_base: Optional API base URL override
            litellm_params: LiteLLM parameters

        Returns:
            Complete URL for the Responses API endpoint
        """
        if api_base is None:
            try:
                api_base = get_chatgpt_oauth_api_base()
            except ChatGPTOAuthError:
                api_base = OPENAI_API_BASE_URL

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Determine if we're using ChatGPT backend or OpenAI API
        is_chatgpt_backend = "backend-api" in api_base or api_base == CHATGPT_BACKEND_BASE_URL

        if is_chatgpt_backend:
            # ChatGPT backend: chatgpt.com/backend-api/codex/responses
            # The base URL already includes the path prefix, just append /responses
            return f"{api_base}/responses"
        else:
            # Standard OpenAI API: api.openai.com/v1/responses
            # Ensure we have /v1 in the path
            if not api_base.endswith("/v1"):
                if "api.openai.com" in api_base and "/v1" not in api_base:
                    api_base = f"{api_base}/v1"
            return f"{api_base}/responses"

    def get_delete_response_url(
        self,
        api_base: Optional[str],
        response_id: str,
        litellm_params: dict,
    ) -> str:
        """
        Get the URL for deleting a response.

        Args:
            api_base: Optional API base URL override
            response_id: The ID of the response to delete
            litellm_params: LiteLLM parameters

        Returns:
            Complete URL for the DELETE response endpoint
        """
        base_url = self.get_complete_url(api_base, litellm_params)
        return f"{base_url}/{response_id}"

    def get_get_response_url(
        self,
        api_base: Optional[str],
        response_id: str,
        litellm_params: dict,
    ) -> str:
        """
        Get the URL for fetching a response by ID.

        Args:
            api_base: Optional API base URL override
            response_id: The ID of the response to fetch
            litellm_params: LiteLLM parameters

        Returns:
            Complete URL for the GET response endpoint
        """
        base_url = self.get_complete_url(api_base, litellm_params)
        return f"{base_url}/{response_id}"
