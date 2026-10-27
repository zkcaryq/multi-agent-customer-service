"""OpenAI 兼容协议的实现。

百炼 compatible-mode、OpenAI、DeepSeek、Moonshot、自建 vLLM 网关都走这一份代码，
换供应商只需要改 CHAT_BASE_URL / CHAT_API_KEY / LLM_MODEL_*。
"""

from __future__ import annotations

import json  # 只用来解析 function.arguments 那段 JSON 字符串
from typing import Any  # client 标注成 Any，避免 openai 的类型出现在签名里

from app.config import settings
from app.providers.base import ChatResult, ProviderError, ToolCall
# 依赖只有配置和协议两处：这份实现不 import openai，SDK 在 client 属性里按需再导

PROVIDER_NAME = "openai_compat"  # 必须与 registry._CHAT 的键一致；CHAT_PROVIDER 默认是 langchain_openai，这份实现是降级/对照路径


class OpenAICompatChat:
    def __init__(self) -> None:
        self._client: Any = None  # 声明成 Any：SDK 类型不能出现在这里，否则上层就依赖 openai 了

    @property
    def client(self) -> Any:
        # 懒建连：registry 构造实例时不碰网络也不查凭据，缺 key 推迟到第一次调用才暴露。
        if self._client is None:
            if not settings.chat_api_key:
                raise ProviderError("未配置 CHAT_API_KEY（或 DASHSCOPE_API_KEY）")
            from openai import AsyncOpenAI  # 函数内导入：没装 openai 时本模块仍可 import；报错会由调用处的 except Exception 包成 ProviderError

            self._client = AsyncOpenAI(
                api_key=settings.chat_api_key,
                base_url=settings.chat_base_url,  # 默认指向百炼 compatible-mode，换成自建 vLLM 只改这一个值
                timeout=settings.chat_timeout_s,  # CHAT_TIMEOUT_S，默认 30 秒；只约束模型调用，与工具执行器 5 秒超时无关
                max_retries=settings.chat_max_retries,  # SDK 内部重试，默认 2 次；和外层 retry_with_backoff 是乘算关系，最坏耗时叠加
            )
        return self._client  # 实例被 registry 按 chat:openai_compat 缓存，所以这个客户端（含连接池）在进程内复用

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
        # 岗位 → 模型 id；这行在 try 之外，未知岗位抛的是 KeyError，不会被归一成 ProviderError
        model = settings.model_for(role)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # 可选参数按需再塞进 kwargs：不传就完全不带该字段，兼容性由网关自己判
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}  # 不支持 JSON 模式的网关会回 400 → ProviderError
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"  # 不是 "required"：模型可以只回文本不调工具，上层不能假设一定有 tool_calls

        try:
            resp = await self.client.chat.completions.create(**kwargs)  # self.client 也在 try 内，缺 key 的 ProviderError 靠下面一条原样透传
        except ProviderError:
            raise  # 拦在前面，否则配置错误会被下一行的宽泛捕获改写成"调用失败"，把 CHAT_API_KEY 的提示文案弄丢
        except Exception as exc:
            # 超时/连接/HTTP 状态码/响应解析异常都归一成 ProviderError；from exc 保留原始 traceback
            # 已知面：exc 原文可能带网关返回内容，这里只写日志，接口层给用户的是 503 固定文案
            raise ProviderError(f"{PROVIDER_NAME} 调用失败 role={role} model={model}: {exc}") from exc

        # 安全拦截或空补全时 choices 可能为空列表；这里不报错，退化成"空内容 + 无工具调用"
        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        return ChatResult(
            content=(msg.content if msg and msg.content else ""),  # None / 空串都归一成 ""，避免下游拿到 None
            tool_calls=_parse_tool_calls(msg),  # msg 为 None 时返回 []，见下面的 getattr 兜底
            model=model,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,  # usage 为 None 时 getattr 也安全返回 0；or 0 兜住 0 值字段缺失
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            provider=PROVIDER_NAME,  # 写死而不是取网关回显，方便和 registry 配置对齐
        )


def _parse_tool_calls(msg: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in getattr(msg, "tool_calls", None) or []:  # msg 为 None 或结构不符都退化成不循环，本函数不抛错
        try:
            args = json.loads(tc.function.arguments or "{}")  # 参数在协议里是 JSON 字符串；空串按 {} 处理
        except json.JSONDecodeError:
            args = {}  # 半截 JSON 丢参数但保留调用，执行器随后报缺必填；比整条丢弃更容易在 trace 里看出来
        # 与 langchain_chat 不同，这里没有过滤脏工具名，也没校验 loads 出来的是不是 dict
        # （模型若吐 `[1,2]` 这类合法 JSON，arguments 就成了 list，下游按 dict 用会出错）
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return calls
