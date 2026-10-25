"""复杂任务评测集的守门测试：结构错了要在跑评测之前就红，而不是跑一半才发现。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "data" / "eval" / "complex_tasks.json"
CASES = json.loads(DATA.read_text(encoding="utf-8"))

EXPECT_KEYS = {
    "tools_all", "tools_any", "tools_none", "tool_success", "tool_success_if_called",
    "arguments", "requires_approval", "approve", "should_handoff", "consultations_any",
    "must_contain_any", "must_not_contain",
    "db_refunds", "db_handoffs_delta", "db_order_status", "db_address_contains",
}


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_case_has_turns_and_category():
    for case in CASES:
        assert case["category"], case["id"]
        assert case["turns"] and all(t.strip() for t in case["turns"]), case["id"]


def test_expect_keys_are_known():
    for case in CASES:
        unknown = set(case["expect"]) - EXPECT_KEYS
        assert not unknown, f"用例 {case['id']} 有未实现的断言字段: {unknown}"


def test_approve_only_when_approval_required():
    for case in CASES:
        expect = case["expect"]
        if "approve" in expect:
            assert expect.get("requires_approval") is True, f"用例 {case['id']} 批准了但没要求审批"


def test_eval_orders_are_isolated():
    """用例只准碰 ORD-EVAL-* 订单，避免污染演示数据与幂等索引。"""
    for case in CASES:
        order_id = case.get("order_id")
        if order_id:
            assert order_id.startswith("ORD-EVAL-"), f"用例 {case['id']} 用了非评测订单 {order_id}"


def test_referenced_tools_are_registered():
    from app.agents import load_all_agents
    from app.tools.registry import tool_registry

    load_all_agents()
    known = {spec.definition.name for spec in tool_registry.list_all()}
    for case in CASES:
        expect = case["expect"]
        named = set(expect.get("tools_all", [])) | set(expect.get("tools_any", [])) \
            | set(expect.get("tools_none", [])) | set(expect.get("tool_success", {})) \
            | set(expect.get("tool_success_if_called", {}))
        assert named <= known, f"用例 {case['id']} 引用了未注册工具: {named - known}"


@pytest.mark.parametrize("case", CASES, ids=[f"case{c['id']}" for c in CASES])
def test_each_case_asserts_something(case):
    """一条用例至少要有一个可判定的断言，否则白送一次通过。"""
    assert set(case["expect"]) - {"tools_none"}, f"用例 {case['id']} 没有任何有效断言"
