"""
ChatGPT OAuth Provider for LiteLLM.

This provider enables using ChatGPT Plus subscription credentials (OAuth access/refresh tokens)
to make API calls through LiteLLM. It supports both the Chat Completions API and the Responses API.

Uses the ChatGPT backend API at chatgpt.com/backend-api/codex with OAuth access tokens.

The provider reads OAuth tokens from:
1. Environment variables (CHATGPT_OAUTH_ACCESS_TOKEN, CHATGPT_OAUTH_REFRESH_TOKEN, etc.)
2. The Codex CLI auth file (~/.codex/auth.json)

Usage:
    # In litellm_config.yaml:
    model_list:
      - model_name: gpt-5.1-codex-max
        litellm_params:
          model: chatgpt_oauth/gpt-5.1-codex-max

    # Or programmatically:
    import litellm
    response = litellm.responses(
        model="chatgpt_oauth/gpt-5.1-codex-max",
        input="Write a function to calculate fibonacci numbers"
    )
"""

from litellm.llms.chatgpt_oauth.common_utils import (
    ChatGPTOAuthError,
    ChatGPTOAuthTokenManager,
    CHATGPT_BACKEND_BASE_URL,
    get_chatgpt_oauth_api_base,
    get_chatgpt_oauth_headers,
    get_token_manager,
)

__all__ = [
    "ChatGPTOAuthError",
    "ChatGPTOAuthTokenManager",
    "CHATGPT_BACKEND_BASE_URL",
    "get_chatgpt_oauth_api_base",
    "get_chatgpt_oauth_headers",
    "get_token_manager",
]
