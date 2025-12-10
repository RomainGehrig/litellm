"""
ChatGPT OAuth Provider for LiteLLM.

This provider enables using ChatGPT Plus subscription credentials (OAuth access/refresh tokens)
to make API calls through LiteLLM. It supports both the Chat Completions API and the Responses API.

Supports two backend modes:
1. ChatGPT Backend mode: Uses OAuth access token directly with chatgpt.com/backend-api/codex
2. Standard OpenAI mode: Exchanges OAuth token for API key and uses api.openai.com/v1

The provider reads OAuth tokens from:
1. Environment variables (CHATGPT_OAUTH_ACCESS_TOKEN, CHATGPT_OAUTH_REFRESH_TOKEN, etc.)
2. The Codex CLI auth file (~/.codex/auth.json)

Configuration:
    Set CHATGPT_OAUTH_MODE to control the backend:
    - "auto" (default): Auto-detect based on available credentials
    - "chatgpt_backend": Force ChatGPT backend mode (requires account_id)
    - "openai_api": Force standard OpenAI API mode (requires token exchange)

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
    ChatGPTBackendMode,
    ChatGPTOAuthError,
    ChatGPTOAuthTokenManager,
    CHATGPT_BACKEND_BASE_URL,
    OPENAI_API_BASE_URL,
    get_chatgpt_oauth_api_base,
    get_chatgpt_oauth_credentials,
    get_chatgpt_oauth_headers,
    get_chatgpt_oauth_mode,
    get_token_manager,
)

__all__ = [
    "ChatGPTBackendMode",
    "ChatGPTOAuthError",
    "ChatGPTOAuthTokenManager",
    "CHATGPT_BACKEND_BASE_URL",
    "OPENAI_API_BASE_URL",
    "get_chatgpt_oauth_api_base",
    "get_chatgpt_oauth_credentials",
    "get_chatgpt_oauth_headers",
    "get_chatgpt_oauth_mode",
    "get_token_manager",
]
