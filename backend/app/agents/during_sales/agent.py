"""During-sales Agent — 下单、支付、改地址、发票。图结构见 base_agent。"""

from __future__ import annotations

from app.agents.base_agent import DomainAgent
from app.agents.during_sales.prompt import DURING_SALES_SYSTEM_PROMPT
from app.agents.during_sales.tools import register_during_sales_tools

_DECISION_PROMPT = (
    "你是售中客服决策助手。根据用户问题和已有知识库内容，判断是否需要调用工具获取实时数据。\n"
    "需要实时数据的场景：查询订单状态、查询支付信息、修改地址、申请发票。\n"
    "不需要工具的场景：支付方式咨询、发票政策说明、地址修改规则说明（知识库已覆盖）。\n"
    "如果需要工具，选择最合适的工具并填写参数。如果不需要，返回空数组。"
)


class DuringSalesAgent(DomainAgent):
    AGENT_ID = 2
    # 岗位标识同时用于 provider 模型配置、日志和图 thread_id；不是单独训练的一套模型。
    AGENT_NAME = "during_sales"
    # 决策提示词只负责选工具；最终自然语言回答用另一个 system prompt。
    DECISION_PROMPT = _DECISION_PROMPT

    def register_tools(self) -> None:
        # 子类仅绑定领域工具，公共图结构由 DomainAgent 复用，减少四域审批逻辑重复。
        register_during_sales_tools(agent_id=self.AGENT_ID)

    def build_system_prompt(self, knowledge: str) -> str:
        return DURING_SALES_SYSTEM_PROMPT.replace("{knowledge_context}", knowledge)


# Singleton
during_sales_agent = DuringSalesAgent()
