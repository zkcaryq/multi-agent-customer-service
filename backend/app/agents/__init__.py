"""Agent package helpers."""

from __future__ import annotations


def load_all_agents() -> None:
    """Construct every domain Agent so the runtime tool list is deterministic."""
    from app.agents.after_sales.agent import after_sales_agent  # noqa: F401
    from app.agents.complaint.agent import complaint_agent  # noqa: F401
    from app.agents.during_sales.agent import during_sales_agent  # noqa: F401
    from app.agents.pre_sales.agent import pre_sales_agent  # noqa: F401
