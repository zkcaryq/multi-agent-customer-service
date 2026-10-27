"""基于 LangChain 的 ChatModel 适配器。

LangChain 负责模型消息、结构化输出和 Tool Calling 协议适配；业务层仍只依赖
``ChatProvider``，工具校验、审批、重试和审计继续由本项目自己控制。
"""

from __future__ import annotations

from typing import Any  # model 变量标注成 Any，不把 LangChain 的 runnable 类型暴露到签名上

from app.config import settings
from app.providers.base import ChatResult, ProviderError, ToolCall  # 只依赖协议层的数据结构


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
        # 这组参数是 ChatProvider 协议的手工镜像（没有继承，改 base.py 不会在这里报错），
        # 全部关键字传参与 openai_compat 那份实现对齐，调用方可以互换。
        # 在出网前显式报配置错误；密钥只用于认证，不应出现在日志、源码注释或任何提交物里。
        if not settings.chat_api_key:
            raise ProviderError("未配置 CHAT_API_KEY（或 DASHSCOPE_API_KEY）")  # 读的是 chat_api_key，它自身已含 DASHSCOPE_API_KEY 回落

        try:
            from langchain_openai import ChatOpenAI  # 函数内导入：没装 langchain-openai 时本模块仍可导入，缺包在这里变成 ProviderError

            # 这里每次 chat 都创建 ChatOpenAI；registry 缓存的是包装实例，不能声称复用了这个模型对象的连接池。
            model: Any = ChatOpenAI(
                # intent/summary/业务域是岗位名，分别取配置；四个 Agent 可以使用同一个底层模型。
                model=settings.model_for(role),
                api_key=settings.chat_api_key,
                base_url=settings.chat_base_url,  # 与 openai_compat 共用同一个 CHAT_BASE_URL，所以换网关只改配置
                # 这是模型请求超时，与 tools.executor 中业务工具的 5 秒超时独立。
                timeout=settings.chat_timeout_s,
                # SDK 的模型调用重试不计入业务工具 CircuitBreaker。
                max_retries=settings.chat_max_retries,
                temperature=temperature,  # 只能在建对象时给：LangChain 把参数烘进实例，不是每次 invoke 传
                max_tokens=max_tokens,  # 同上；换 temperature 需要重新构造对象，这正是每次新建的代价
                # 供应商扩展参数，换兼容网关时仍需核对支持情况；统一接口不代表所有参数都通用。
                extra_body={"enable_thinking": settings.chat_enable_thinking},  # CHAT_ENABLE_THINKING 默认 False：客服链路优先低延迟；非百炼网关可能直接拒绝这个字段
            )
            if json_mode:
                # bind 返回绑定参数的 runnable，不是把返回值验证成 IntentResult。
                model = model.bind(response_format={"type": "json_object"})  # 覆盖 model 变量，后续出错时仍用 role 反查模型名，不受 bind 影响
            if tools:
                # 只把工具描述交给模型；LangChain 此调用不执行 handler，是否调用由模型选择。
                model = model.bind_tools(tools, tool_choice="auto")  # "auto" 允许模型不调工具，与 openai_compat 的取值一致

            # 异步等待完整模型响应，当前不是 astream/token 流；WebSocket 后续只是切块展示。
            response = await model.ainvoke(messages)  # 超时/断连由 LangChain 抛出，统一被下面的 except Exception 归一
        except ProviderError:
            raise  # 与 chat.py 同样的写法，此处偏防御性：try 块内部目前不会主动抛 ProviderError
        except Exception as exc:
            raise ProviderError(
                f"langchain_openai 调用失败 role={role} model={settings.model_for(role)}: {exc}"
            ) from exc  # 错误文案里再调一次 model_for：若 role 是未知岗位，这里会抛 KeyError 顶替原始异常（前面已用过同一调用，实际到不了这个分支）

        # 读取这一次模型响应的用量，缺失按 0；不是整个 Agent 链路的累加统计。
        usage = getattr(response, "usage_metadata", None) or {}  # LangChain 用 usage_metadata，字段名与 OpenAI 的 usage 不同，这里按 LangChain 形状取
        tool_calls = []  # 先建空列表：模型只回文本时 ChatResult.tool_calls 就是空，执行器据此判定"本轮无工具"
        # LangChain 已把供应商 function arguments 解析成统一结构，业务层无需再次 json.loads。
        for call in getattr(response, "tool_calls", None) or []:  # 不同 LangChain 版本该字段可能是 dict 或对象，取不到就当没有
            # 有的模型会把 name 返回成 [] / null 这类脏值；不拦掉就会被当成一个叫
            # "[]" 的工具执行失败，污染 trace 并可能误触发"工具全失败转人工"
            name = call.get("name")
            # 先过滤明显坏工具名；是否注册和是否允许当前 Agent 执行是另外的校验问题。
            if not isinstance(name, str) or not name.strip():
                continue  # 静默丢弃整个调用：模型想调工具但名字坏掉时，上层只会看到"没有 tool_calls"，直接进入文本回复
            tool_calls.append(ToolCall(
                id=str(call.get("id") or ""),  # 缺 id 时给空串而不是造一个：回填工具结果时对齐靠它，LangChain 侧多数场景不强依赖
                name=name,
                # 非对象参数退为空字典，执行器稍后会报告缺必填；不是自动修复缺失的业务参数。
                arguments=call.get("args") if isinstance(call.get("args"), dict) else {},
            ))

        content = response.content  # 类型取决于模型与是否流式：可能是 str，也可能是内容块列表
        # 兼容分块内容列表，只拼 text 字段；这里不实现多模态输出渲染。
        if isinstance(content, list):
            content = "".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)  # 非 dict 块整个 str() 化，可能把 repr 混进正文；思考内容等其它键被丢掉
                for block in content
            )

        # 统一供应商返回格式，让 DomainAgent 不依赖 AIMessage 等框架类型。
        return ChatResult(
            content=str(content or ""),  # 再套一层 str() 兜住 None / 非字符串，保证下游拿到的一定是 str
            tool_calls=tool_calls,
            # intent/summary/业务域是岗位名，分别取配置；四个 Agent 可以使用同一个底层模型。
            model=settings.model_for(role),  # 第三次调用 model_for：取的是配置值而不是响应回显，网关换模型时这里不会跟着变
            prompt_tokens=int(usage.get("input_tokens") or 0),  # LangChain 命名 input/output，与 OpenAI 的 prompt/completion 对应
            completion_tokens=int(usage.get("output_tokens") or 0),
            provider="langchain_openai",  # 写死，必须与 registry._CHAT 的键一致，trace 与自检靠它区分两份实现
        )
