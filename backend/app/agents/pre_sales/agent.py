"""Pre-sales Agent — 商品咨询、规格、库存、优惠、会员。图结构见 base_agent。"""

from __future__ import annotations

from app.agents.base_agent import DomainAgent
from app.agents.pre_sales.prompt import PRE_SALES_SYSTEM_PROMPT
from app.agents.pre_sales.tools import register_pre_sales_tools

_DECISION_PROMPT = (
    "你是售前客服决策助手。根据用户问题和已有知识库内容，判断是否需要调用工具获取实时数据。\n"
    "需要实时数据的场景：库存查询、价格查询、优惠券查询、会员信息查询。\n"
    "不需要工具的场景：一般商品咨询、功能介绍、对比说明（知识库已覆盖）。\n"
    "如果需要工具，选择最合适的工具并填写参数。如果不需要，返回空数组。"
)


class PreSalesAgent(DomainAgent):
    AGENT_ID = 1
    # 岗位标识同时用于 provider 模型配置、日志和图 thread_id；不是单独训练的一套模型。
    AGENT_NAME = "pre_sales"
    # 决策提示词只负责选工具；最终自然语言回答用另一个 system prompt。
    DECISION_PROMPT = _DECISION_PROMPT

    def register_tools(self) -> None:
        # 子类仅绑定领域工具，公共图结构由 DomainAgent 复用，减少四域审批逻辑重复。
        register_pre_sales_tools(agent_id=self.AGENT_ID)

    def build_system_prompt(self, knowledge: str) -> str:
        return PRE_SALES_SYSTEM_PROMPT.replace("{knowledge_context}", knowledge)


# Singleton
pre_sales_agent = PreSalesAgent()
