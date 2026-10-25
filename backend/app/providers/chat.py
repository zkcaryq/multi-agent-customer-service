"""OpenAI 兼容协议的实现。

百炼 compatible-mode、OpenAI、DeepSeek、Moonshot、自建 vLLM 网关都走这一份代码，
换供应商只需要改 CHAT_BASE_URL / CHAT_API_KEY / LLM_MODEL_*。
"""

from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.providers.base import ChatResult, ProviderError, ToolCall

PROVIDER_NAME = "openai_compat"


class OpenAICompatChat:
    def __init__(self) -> None:
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not settings.chat_api_key:
                raise ProviderError("未配置 CHAT_API_KEY（或 DASHSCOPE_API_KEY）")
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.chat_api_key,
                base_url=settings.chat_base_url,
                timeout=settings.chat_timeout_s,
                max_retries=settings.chat_max_retries,
            )
        return self._client

    async def chat(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        model = settings.model_for(role)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{PROVIDER_NAME} 调用失败 role={role} model={model}: {exc}") from exc

        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        return ChatResult(
            content=(msg.content if msg and msg.content else ""),
            tool_calls=_parse_tool_calls(msg),
            model=model,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            provider=PROVIDER_NAME,
        )


def _parse_tool_calls(msg: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return calls
