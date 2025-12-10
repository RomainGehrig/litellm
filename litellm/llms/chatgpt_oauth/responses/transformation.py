"""
ChatGPT OAuth Responses API transformation.

This module provides the configuration class for making Responses API requests
(OpenAI's newer API for GPT-5/Codex models) using ChatGPT Plus OAuth credentials.
"""

from typing import Optional

from litellm._logging import verbose_logger
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import get_chatgpt_oauth_credentials, ChatGPTOAuthError


class ChatGPTOAuthResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for ChatGPT OAuth Responses API.

    This class extends OpenAIResponsesAPIConfig to use OAuth tokens from ChatGPT Plus
    instead of a static API key. It's designed for GPT-5/Codex models that use the
    newer Responses API endpoint (/v1/responses).

    The class handles:
    - Reading OAuth tokens from ~/.codex/auth.json or environment variables
    - Automatic token refresh when tokens expire
    - Token exchange to get API keys when needed
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

        This method gets the OAuth token and sets up the Authorization header.

        Args:
            headers: Request headers dict to update
            model: Model name
            litellm_params: LiteLLM specific parameters

        Returns:
            Updated headers dict with Authorization header
        """
        try:
            _, oauth_token = get_chatgpt_oauth_credentials()

            headers.update({
                "Authorization": f"Bearer {oauth_token}",
            })

            # Ensure Content-Type is set
            if "Content-Type" not in headers:
                headers["Content-Type"] = "application/json"

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

        Args:
            api_base: Optional API base URL override
            litellm_params: LiteLLM parameters

        Returns:
            Complete URL for the Responses API endpoint
        """
        if api_base is None:
            try:
                api_base, _ = get_chatgpt_oauth_credentials()
            except ChatGPTOAuthError:
                api_base = "https://api.openai.com/v1"

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Ensure we're using the /v1 base, not the root
        if not api_base.endswith("/v1"):
            if "api.openai.com" in api_base and "/v1" not in api_base:
                api_base = f"{api_base}/v1"

        return f"{api_base}/responses"
