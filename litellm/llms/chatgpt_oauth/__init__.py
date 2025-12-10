"""
ChatGPT OAuth Provider for LiteLLM.

This provider enables using ChatGPT Plus subscription credentials (OAuth access/refresh tokens)
to make API calls through LiteLLM. It supports both the Chat Completions API and the Responses API.

The provider reads OAuth tokens from:
1. Environment variables (CHATGPT_OAUTH_ACCESS_TOKEN, CHATGPT_OAUTH_REFRESH_TOKEN, etc.)
2. The Codex CLI auth file (~/.codex/auth.json)

Usage:
    # In litellm_config.yaml:
    model_list:
      - model_name: gpt-5.1-codex-max
        litellm_params:
          model: chatgpt_oauth/gpt-5.1-codex-max
          api_base: https://api.openai.com/v1

    # Or programmatically:
    import litellm
    response = litellm.completion(
        model="chatgpt_oauth/gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

from litellm.llms.chatgpt_oauth.common_utils import (
    ChatGPTOAuthError,
    ChatGPTOAuthTokenManager,
    get_chatgpt_oauth_credentials,
    get_token_manager,
)

__all__ = [
    "ChatGPTOAuthError",
    "ChatGPTOAuthTokenManager",
    "get_chatgpt_oauth_credentials",
    "get_token_manager",
]
