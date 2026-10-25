"""Multi-Agent 协调器：统一完成意图识别、领域路由和执行留痕。

【为什么需要这一层】
如果让 `api/chat.py` 直接做「分类 → 挑 Agent → 并行咨询 → 调 run」，接口层会同时
承担 HTTP 契约和编排逻辑，换 UI（Widget / 后台测试页 / WebSocket）就要复制一遍。
所以把编排抽成 Supervisor：

    api/chat.py（HTTP 契约）
      → coordinator.plan()   决定「谁主、谁辅」
      → coordinator.run()    并行咨询 + 主 Agent 执行 + 写 trace
        → DomainAgent.run()  领域内的 LangGraph（检索/工具/审批/生成）

【模式：Supervisor + 专业领域 Agent】
协调器**不直接办理业务**，也不碰工具；它只做三件事：
1. 选主 Agent（LLM 分类 + 关键词兜底 + 确定性守卫）；
2. 选辅助 Agent（关键词打分，最多 max_collaboration_agents 个）；
3. 让辅助 Agent 以**只读顾问**身份并行给建议，再把建议注入主 Agent 的生成阶段。

【关键设计：辅助 Agent 只读】
consult() 不调工具、不写库。否则"查物流顺带问优惠券"会同时产生退款/建单等副作用，
一次提问变两笔业务。需要动业务的操作只能由主 Agent 在自己的图里做（且高危要审批）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.agents.router import (
    IntentResult,          # 分类结果：意图名 / 目标 Agent / 置信度 / 分类器
    agent_by_name,         # 名字 → Agent 单例（审批回调也用它找回线程）
    classify_intent,       # LLM 优先、关键词兜底
    collaboration_candidates,  # 关键词打分挑辅助领域
    route_to_agent,        # IntentResult → Agent 实例
)
from app.config import settings


@dataclass
class CoordinationPlan:
    """一次请求的"编队方案"，在 plan() 产生、run() 消费。

    把方案单独拿出来（而不是在 run 里现算）的好处：
    接口层可以在真正执行前先看一眼编队（日志/评测/调试），也便于单测只测路由。
    """
    intent: IntentResult     # 主意图（含目标 Agent、置信度、分类器来源）
    agent: Any               # 目标 Agent 实例（主 Agent）
    participants: list[str]  # 参与的领域名，主 Agent 恒排第一


class MultiAgentCoordinator:
    """面向 API 的 Supervisor，避免路由和执行逻辑散落在接口层。"""

    async def plan(self, user_query: str) -> CoordinationPlan:
        """只做"分诊"，不产生任何副作用，也不调模型两次。

        分类只调一次模型（intent 岗位）；辅助领域的挑选用确定性关键词打分，
        不再额外请求模型——多一次模型调用就多一份延迟和不确定性。
        """
        intent = await classify_intent(user_query)
        participants = collaboration_candidates(
            user_query,
            intent.target_agent,
            limit=settings.max_collaboration_agents,   # 默认最多 2 个领域（主 + 1 辅）
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
        """并行咨询辅助 Agent，再交给主 Agent 走完整状态机。"""
        # 排除主 Agent：participants 里第一个就是它，不需要"自己咨询自己"
        consultant_names = [name for name in plan.participants if name != plan.intent.target_agent]
        consultations: list[dict[str, Any]] = []
        if consultant_names:
            # asyncio.gather 并发：两个咨询互不依赖，串行会让首字延迟翻倍。
            # return_exceptions=True 是关键：一个辅助 Agent 挂了不能让整条请求 500，
            # 失败的那个会在下面被转换成一个"不可用"的建议条目。
            outcomes = await asyncio.gather(
                *(agent_by_name(name).consult(user_query) for name in consultant_names),
                return_exceptions=True,
            )
            for name, outcome in zip(consultant_names, outcomes):
                if isinstance(outcome, Exception):
                    # 降级：把异常翻译成给主 Agent 看的一句话，而不是抛给用户
                    consultations.append({
                        "agent": name,
                        "content": "辅助 Agent 暂时不可用，主 Agent 独立完成处理。",
                        "sources": [],
                        "error": str(outcome),
                    })
                else:
                    consultations.append(outcome)

        # 主 Agent 执行：consultations 会作为 collaboration_context 进入生成阶段，
        # 由 _generate 拼成一条 system 消息（并明确"辅助 Agent 没有执行任何操作"）
        state = await plan.agent.run(
            conversation_id=conversation_id,
            user_query=user_query,
            user_id=user_id,
            collaboration_context=consultations,
        )

        # 把"这次是怎么协同的"写进 trace，供后台监控与离线评测复盘。
        # 注意只存摘要（agent/sources/success），不存咨询正文，避免 trace 过肥。
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


# 模块级单例：协调器无状态（状态都在 Agent/checkpointer 里），可以全局复用
coordinator = MultiAgentCoordinator()
