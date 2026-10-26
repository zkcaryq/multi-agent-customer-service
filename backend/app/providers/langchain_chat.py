"""基于 LangChain 的 ChatModel 适配器。

LangChain 负责模型消息、结构化输出和 Tool Calling 协议适配；业务层仍只依赖
``ChatProvider``，工具校验、审批、重试和审计继续由本项目自己控制。
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.providers.base import ChatResult, ProviderError, ToolCall


class LangChainChat:
    """通过 ``langchain-openai`` 调用任意 OpenAI 兼容模型端点。"""

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
        if not settings.chat_api_key:
            raise ProviderError("未配置 CHAT_API_KEY（或 DASHSCOPE_API_KEY）")

        try:
            from langchain_openai import ChatOpenAI

            model: Any = ChatOpenAI(
                model=settings.model_for(role),
                api_key=settings.chat_api_key,
                base_url=settings.chat_base_url,
                timeout=settings.chat_timeout_s,
                max_retries=settings.chat_max_retries,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": settings.chat_enable_thinking},
            )
            if json_mode:
                model = model.bind(response_format={"type": "json_object"})
            if tools:
                model = model.bind_tools(tools, tool_choice="auto")

            response = await model.ainvoke(messages)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"langchain_openai 调用失败 role={role} model={settings.model_for(role)}: {exc}"
            ) from exc

        usage = getattr(response, "usage_metadata", None) or {}
        tool_calls = []
        for call in getattr(response, "tool_calls", None) or []:
            # 有的模型会把 name 返回成 [] / null 这类脏值；不拦掉就会被当成一个叫
            # "[]" 的工具执行失败，污染 trace 并可能误触发"工具全失败转人工"
            name = call.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_calls.append(ToolCall(
                id=str(call.get("id") or ""),
                name=name,
                arguments=call.get("args") if isinstance(call.get("args"), dict) else {},
            ))

        content = response.content
        if isinstance(content, list):
            content = "".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
            )

        return ChatResult(
            content=str(content or ""),
            tool_calls=tool_calls,
            model=settings.model_for(role),
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
            provider="langchain_openai",
        )
