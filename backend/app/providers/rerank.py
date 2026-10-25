"""Rerank 供应商实现。

百炼 rerank 不在 OpenAI 兼容路径上，是 DashScope 原生端点，所以单独走 httpx。
`noop` 表示不做重排，由上层保留融合分数原序（离线/降级用）。
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.providers.base import ProviderError

PROVIDER_NAME_DASHSCOPE = "dashscope"
PROVIDER_NAME_NOOP = "noop"


class DashScopeRerank:
    name = PROVIDER_NAME_DASHSCOPE

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []

        api_key = settings.key_for("rerank")
        if not api_key:
            raise ProviderError("未配置 RERANK_API_KEY / CHAT_API_KEY")

        import httpx

        payload: dict[str, Any] = {
            "model": settings.rerank_model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
        try:
            async with httpx.AsyncClient(timeout=settings.rerank_timeout_s) as client:
                resp = await client.post(
                    settings.rerank_endpoint,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()
        except Exception as exc:
            raise ProviderError(
                f"rerank 调用失败 model={settings.rerank_model}: {exc}"
            ) from exc

        results = (body.get("output") or {}).get("results") or []
        return [
            (int(r["index"]), float(r.get("relevance_score", 0.0)))
            for r in results
            if "index" in r
        ]


class NoopRerank:
    name = PROVIDER_NAME_NOOP

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        return []
