"""Multi-Agent 协调器：统一完成意图识别、领域路由和执行留痕。

这里采用 Supervisor + 专业领域 Agent 模式。协调器不直接办理业务，只选择售前、
售中、售后或投诉 Agent；领域 Agent 再通过 LangGraph 完成检索、工具和审批流程。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.agents.router import (
    IntentResult,
    agent_by_name,
    classify_intent,
    collaboration_candidates,
    route_to_agent,
)
from app.config import settings


@dataclass
class CoordinationPlan:
    intent: IntentResult
    agent: Any
    participants: list[str]


class MultiAgentCoordinator:
    """面向 API 的 Supervisor，避免路由和执行逻辑散落在接口层。"""

    async def plan(self, user_query: str) -> CoordinationPlan:
        intent = await classify_intent(user_query)
        participants = collaboration_candidates(
            user_query,
            intent.target_agent,
            limit=settings.max_collaboration_agents,
        )
        return CoordinationPlan(
            intent=intent,
            agent=route_to_agent(intent),
            participants=participants,
        )

    async def run(
        self,
        plan: CoordinationPlan,
        *,
        conversation_id: int,
        user_query: str,
        user_id: str,
    ) -> dict[str, Any]:
        consultant_names = [name for name in plan.participants if name != plan.intent.target_agent]
        consultations: list[dict[str, Any]] = []
        if consultant_names:
            outcomes = await asyncio.gather(
                *(agent_by_name(name).consult(user_query) for name in consultant_names),
                return_exceptions=True,
            )
            for name, outcome in zip(consultant_names, outcomes):
                if isinstance(outcome, Exception):
                    consultations.append({
                        "agent": name,
                        "content": "辅助 Agent 暂时不可用，主 Agent 独立完成处理。",
                        "sources": [],
                        "error": str(outcome),
                    })
                else:
                    consultations.append(outcome)

        state = await plan.agent.run(
            conversation_id=conversation_id,
            user_query=user_query,
            user_id=user_id,
            collaboration_context=consultations,
        )
        state.setdefault("trace", {})["coordination"] = {
            "pattern": "supervisor_with_parallel_consultation",
            "selected_agent": plan.intent.target_agent,
            "participants": plan.participants,
            "intent": plan.intent.intent_name,
            "confidence": plan.intent.confidence,
            "classifier": plan.intent.classifier,
            "consultations": [
                {
                    "agent": item["agent"],
                    "sources": item.get("sources", []),
                    "success": not bool(item.get("error")),
                }
                for item in consultations
            ],
        }
        return state


coordinator = MultiAgentCoordinator()
