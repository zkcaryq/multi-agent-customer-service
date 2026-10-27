"""After-sales Agent — 物流、订单、退款。图结构见 app/agents/base_agent.py。"""

from __future__ import annotations

from app.agents.after_sales.prompt import AFTER_SALES_SYSTEM_PROMPT
from app.agents.after_sales.tools import register_after_sales_tools
from app.agents.base_agent import DomainAgent

_DECISION_PROMPT = (
    "你是售后客服决策助手。根据用户问题和已有知识库内容，判断是否需要调用工具获取实时数据。\n"
    "需要实时数据的场景：查询物流信息、查询订单详情、发起退款、查询退款进度、查询客户画像。\n"
    "不需要工具的场景：退换货政策说明、投诉流程咨询（知识库已覆盖）。\n"
    "如果需要工具，选择最合适的工具并填写参数。如果不需要，返回空数组。"
)


class AfterSalesAgent(DomainAgent):
    AGENT_ID = 3
    # 岗位标识同时用于 provider 模型配置、日志和图 thread_id；不是单独训练的一套模型。
    AGENT_NAME = "after_sales"
    # 决策提示词只负责选工具；最终自然语言回答用另一个 system prompt。
    DECISION_PROMPT = _DECISION_PROMPT

    def register_tools(self) -> None:
        # 子类仅绑定领域工具，公共图结构由 DomainAgent 复用，减少四域审批逻辑重复。
        register_after_sales_tools(agent_id=self.AGENT_ID)

    def build_system_prompt(self, knowledge: str) -> str:
        return AFTER_SALES_SYSTEM_PROMPT.replace("{knowledge_context}", knowledge)


# Singleton
after_sales_agent = AfterSalesAgent()
