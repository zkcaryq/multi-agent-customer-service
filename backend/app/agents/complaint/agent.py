"""Complaint Agent — 投诉受理、定级、建单、进度跟进。"""

from __future__ import annotations

from app.agents.base_agent import AgentState, DomainAgent
from app.agents.complaint.prompt import COMPLAINT_SYSTEM_PROMPT
from app.agents.complaint.tools import register_complaint_tools

_DECISION_PROMPT = (
    "你是投诉受理决策助手。根据用户诉求判断需要哪些实时数据。\n"
    "需要工具的场景：首次受理投诉（创建投诉工单）、用户询问之前的投诉进展（查询我的投诉工单）、"
    "需要核对订单信息（查询客户画像/订单）。\n"
    "不需要工具的场景：政策解释、单纯安抚回复。\n"
    "如果不需要工具，返回空数组。"
)


class ComplaintAgent(DomainAgent):
    AGENT_ID = 4
    # 岗位标识同时用于 provider 模型配置、日志和图 thread_id；不是单独训练的一套模型。
    AGENT_NAME = "complaint"
    # 决策提示词只负责选工具；最终自然语言回答用另一个 system prompt。
    DECISION_PROMPT = _DECISION_PROMPT

    def register_tools(self) -> None:
        # 子类仅绑定领域工具，公共图结构由 DomainAgent 复用，减少四域审批逻辑重复。
        register_complaint_tools(agent_id=self.AGENT_ID)

    def build_system_prompt(self, knowledge: str) -> str:
        return COMPLAINT_SYSTEM_PROMPT.replace("{knowledge_context}", knowledge)

    def _detect_handoff(self, state: AgentState) -> None:
        """建单成功即视为已转人工跟进，比关键词判定可靠。"""
        super()._detect_handoff(state)
        if any(tr.get("tool") == "create_service_ticket" and tr.get("success")
               for tr in state["tool_results"]):
            state["handoff"] = True
            state["handoff_reason"] = "complaint_ticket_created"


# Singleton
complaint_agent = ComplaintAgent()
