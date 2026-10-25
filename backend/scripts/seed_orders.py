"""造演示业务数据：用户 + 订单 + 商品行 + 物流。

幂等：按 order_no / users.external_id upsert，可反复执行。
时间用相对 now 生成，保证"7 天无理由 / 15 天质量"两条策略分支始终可演示。

跑法：backend 目录下 `python scripts/seed_orders.py`
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.models.database import Order, OrderItem, User, UserAttribute, UserTag  # noqa: E402
from app.models.engine import async_session_factory, init_db  # noqa: E402


def _naive_utc() -> datetime:
    """本库 datetime 列是 naive TIMESTAMP，统一写 naive UTC。"""
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


# (order_no, external_id, nickname, status, [(sku, qty, price)], 物流)
ORDERS = [
    (
        "ORD-AIRPODS-PAID", "user_001", "张三", "shipped",
        [("AirPods Pro 2", 1, 1899.00)],
        {
            "carrier": "顺丰速运", "tracking_no": "SF1234567890123",
            "logistics_status": "运输中", "eta_days": 3,
            "events": [("已揽收", "上海市", 3), ("运输中", "上海 → 北京", 2),
                       ("到达分拣中心", "北京朝阳", 1)],
        },
    ),
    (
        # 签收 20 天前：超 7 天无理由窗口，用来演示退款被拒
        "ORD-CASE-DELIVERED", "user_001", "张三", "completed",
        [("手机壳 MagSafe", 1, 299.00)],
        {
            "carrier": "中通快递", "tracking_no": "ZT9876543210987",
            "logistics_status": "已签收", "eta_days": -1,
            "events": [("已揽收", "广州市", 22), ("运输中", "广州 → 北京", 20),
                       ("已签收", "北京", 20)],
        },
    ),
    (
        "ORD-IPHONE-UNPAID", "user_001", "张三", "unpaid",
        [("iPhone 15 Pro 256GB", 1, 8999.00)],
        None,
    ),
    (
        # 属于另一个账号，用来验证越权查询被拦住
        "ORD-OTHER-USER", "demo_user2", "李四", "paid",
        [("机械键盘 87 键", 1, 499.00)],
        None,
    ),
]


# CRM 画像：等级/积分/常用入口 + 标签
CRM_PROFILES = {
    "user_001": (
        {"member_level": "PLUS会员", "points": "2680", "preferred_channel": "APP"},
        ["高价值", "3C偏好"],
    ),
    "demo_user2": (
        {"member_level": "普通", "points": "120", "preferred_channel": "小程序"},
        ["新客"],
    ),
}


async def seed() -> None:
    await init_db()

    now = _naive_utc()
    async with async_session_factory() as session:
        wanted_users = {o[1]: o[2] for o in ORDERS}
        user_ids: dict[str, int] = {}
        for ext, nick in wanted_users.items():
            user = (await session.execute(
                select(User).where(User.external_id == ext)
            )).scalar_one_or_none()
            if user is None:
                user = User(external_id=ext, nickname=nick, phone="")
                session.add(user)
                await session.flush()
            user_ids[ext] = user.id

        # 画像属性与标签按 key upsert，可反复执行
        for ext, (attrs, tags) in CRM_PROFILES.items():
            uid = user_ids[ext]
            existing_attrs = {
                a.attr_key: a for a in (await session.execute(
                    select(UserAttribute).where(UserAttribute.user_id == uid)
                )).scalars().all()
            }
            for key, value in attrs.items():
                if key in existing_attrs:
                    existing_attrs[key].attr_value = value
                else:
                    session.add(UserAttribute(user_id=uid, attr_key=key, attr_value=value))

            existing_tags = {
                t.tag for t in (await session.execute(
                    select(UserTag).where(UserTag.user_id == uid)
                )).scalars().all()
            }
            for tag in tags:
                if tag not in existing_tags:
                    session.add(UserTag(user_id=uid, tag=tag))

        for order_no, ext, _nick, status, items, logistics in ORDERS:
            order = (await session.execute(
                select(Order).where(Order.order_no == order_no)
            )).scalar_one_or_none()
            if order is None:
                order = Order(order_no=order_no)
                session.add(order)

            order.user_id = user_ids[ext]
            order.status = status
            order.amount = sum(q * p for _n, q, p in items)
            order.shipping_address = "北京市朝阳区 XXX 路 100 号" if ext == "user_001" else "上海市浦东新区 YYY 路 1 号"
            order.created_at = now - timedelta(days=25)
            order.paid_at = None if status == "unpaid" else now - timedelta(days=24)
            order.shipped_at = None if status in ("unpaid", "paid") else now - timedelta(days=3)
            order.completed_at = now - timedelta(days=20) if status == "completed" else None
            order.payment_method = "微信支付" if status != "unpaid" else None

            if logistics:
                order.carrier = logistics["carrier"]
                order.tracking_no = logistics["tracking_no"]
                order.logistics_status = logistics["logistics_status"]
                order.estimated_delivery = (now + timedelta(days=logistics["eta_days"])).strftime("%Y-%m-%d")
                order.logistics_events = json.dumps(
                    [{"time": (now - timedelta(days=d)).strftime("%Y-%m-%d %H:%M"),
                      "status": s, "location": loc} for s, loc, d in logistics["events"]],
                    ensure_ascii=False,
                )
            else:
                order.carrier = order.tracking_no = order.logistics_status = None
                order.estimated_delivery = order.logistics_events = None

            # 商品行整体重建，保持与定义一致
            for old in list(order.items):
                await session.delete(old)
            await session.flush()
            for name, qty, price in items:
                session.add(OrderItem(order_id=order.id, sku_name=name, qty=qty, price=price))

        await session.commit()

        rows = (await session.execute(select(Order.order_no, Order.status, Order.amount))).all()
        print(f"seeded orders: {len(rows)}")
        for r in rows:
            print(f"  {r[0]:22s} {r[1]:10s} {r[2]}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(seed()))
