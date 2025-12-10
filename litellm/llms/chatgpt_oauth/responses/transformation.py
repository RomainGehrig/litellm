"""
ChatGPT OAuth Responses API transformation.

This module provides the configuration class for making Responses API requests
(OpenAI's newer API for GPT-5/Codex models) using ChatGPT Plus OAuth credentials.

Uses the ChatGPT backend API at chatgpt.com/backend-api/codex/responses with OAuth access tokens.
"""

from typing import Optional

from litellm._logging import verbose_logger
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import (
    ChatGPTOAuthError,
    CHATGPT_BACKEND_BASE_URL,
    get_chatgpt_oauth_api_base,
    get_chatgpt_oauth_headers,
)


class ChatGPTOAuthResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for ChatGPT OAuth Responses API.

    This class extends OpenAIResponsesAPIConfig to use OAuth tokens from ChatGPT Plus
    instead of a static API key. It's designed for GPT-5/Codex models that use the
    newer Responses API endpoint (/responses).

    Uses the ChatGPT backend at chatgpt.com/backend-api/codex/responses.

    The class handles:
    - Reading OAuth tokens from ~/.codex/auth.json or environment variables
    - Automatic token refresh when tokens expire
    - Providing ChatGPT-Account-ID header for the ChatGPT backend
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

        This method gets the OAuth credentials and sets up the appropriate headers:
        - Authorization header with the OAuth access token
        - ChatGPT-Account-ID header for the ChatGPT backend

        Args:
            headers: Request headers dict to update
            model: Model name
            litellm_params: LiteLLM specific parameters

        Returns:
            Updated headers dict with Authorization and ChatGPT-Account-ID headers
        """
        try:
            oauth_headers = get_chatgpt_oauth_headers()
            headers.update(oauth_headers)

            verbose_logger.debug("ChatGPT OAuth Responses: Headers configured for ChatGPT backend")

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

        URL structure: chatgpt.com/backend-api/codex/responses

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
                api_base = CHATGPT_BACKEND_BASE_URL

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Append /responses endpoint
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
