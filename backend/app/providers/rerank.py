"""Rerank 供应商实现。

百炼 rerank 不在 OpenAI 兼容路径上，是 DashScope 原生端点，所以单独走 httpx。
`noop` 表示不做重排，由上层按融合分数加轻微惩罚排序（离线/降级用）。
"""

from __future__ import annotations

from typing import Any  # payload 用裸 dict 描述，不引 DashScope SDK

from app.config import settings
from app.providers.base import ProviderError

PROVIDER_NAME_DASHSCOPE = "dashscope"  # 与 registry._RERANK 的键、RERANK_PROVIDER 默认值一致
PROVIDER_NAME_NOOP = "noop"  # 关掉重排时配这个值，装配仍然成功，只是永远走降级排序


class DashScopeRerank:
    name = PROVIDER_NAME_DASHSCOPE  # 协议要求的属性，仅自述；没有任何调用方读它

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        # 空候选直接返回空列表，省一次出网；上层 Reranker 更早就 return 了，所以这里只是保险
        if not documents:
            return []

        api_key = settings.key_for("rerank")  # 会回落到 chat 的凭据（百炼一把 key 通吃），但端点不会回落——非阿里网关必须显式配 RERANK_ENDPOINT
        if not api_key:
            raise ProviderError("未配置 RERANK_API_KEY / CHAT_API_KEY")  # 抛在这里而不是启动时：preflight 已检查，但只覆盖配置非空的场景

        import httpx  # 函数内导入：不用 rerank 的部署不必装 httpx

        payload: dict[str, Any] = {
            "model": settings.rerank_model,  # 默认 qwen3-rerank，是 DashScope 模型名，不是 OpenAI 兼容路径的模型名
            # 重排同时看问题和每条候选正文，和 embedding 独立编码文本的用途不同。
            "input": {"query": query, "documents": documents},
            # 只要候选下标与分数，正文已在本地，避免重复传回。
            "parameters": {"top_n": top_n, "return_documents": False},  # top_n 由调用方传 settings.rerank_top_n（默认 5），候选数少于它时按实际条数返回
        }
        try:
            # 每次请求临时创建异步 HTTP 客户端，退出时关闭；不是全局常驻连接池。
            async with httpx.AsyncClient(timeout=settings.rerank_timeout_s) as client:  # RERANK_TIMEOUT_S 默认 10 秒；这个值不随 registry 缓存复用，每次调用都重新握手
                resp = await client.post(
                    settings.rerank_endpoint,  # 原生端点，写死在配置默认值里；换供应商不是改 base_url 就能对上
                    headers={"Authorization": f"Bearer {api_key}"},  # 凭据只进请求头；异常文案里的 httpx 信息一般含 URL 不含 header，但不要把它回显给终端用户
                    json=payload,
                )
                # 把 HTTP 错误转异常，随后包装 ProviderError，让业务层触发降级。
                resp.raise_for_status()
                body = resp.json()  # 网关回 HTML（例如 502 页面）时这里抛 JSONDecodeError，同样落到下面的 ProviderError
        except Exception as exc:
            raise ProviderError(
                f"rerank 调用失败 model={settings.rerank_model}: {exc}"
            ) from exc  # 宽泛捕获：超时/连接/状态码/非 JSON 都归一，由 Reranker 捕获后走 _score_rerank 降级

        # 正常响应缺 results 时视为空；后续类型解析异常在 try 外，不一定被包装成 ProviderError。
        results = (body.get("output") or {}).get("results") or []  # 只认 DashScope 的 output.results 形状；返回顶层字段的其他网关会静默得到空列表，看起来像"provider 正常但没结果"，实际是配置错了
        return [
            (int(r["index"]), float(r.get("relevance_score", 0.0)))  # index 是相对本次 documents 的下标；非数字串会抛 ValueError，冒到只 except ProviderError 的 Reranker（接口层变 500 而不是降级）
            for r in results
            if "index" in r  # 缺 index 的条目直接丢弃，不报错；分数缺失按 0.0 补，会排到最后
        ]


class NoopRerank:
    name = PROVIDER_NAME_NOOP  # 与 _RERANK 表键对应；配置写成别的字符串会在装配期 ProviderError

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        # 固定空列表：调用方把空 ranked 当"需要降级"，于是按融合分数加惩罚排序。
        # 已知限制：这与"真实 provider 挂了"在最终结果上无法区分，只有日志里后者有 warning。
        return []
