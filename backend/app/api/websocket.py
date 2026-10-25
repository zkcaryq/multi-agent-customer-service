"""WebSocket endpoint for Widget long connection with streaming output."""

import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.chat import chat
from app.schemas.chat import ChatRequest

router = APIRouter(tags=["websocket"])


@router.websocket("/chat")
async def websocket_chat(ws: WebSocket) -> None:
    await ws.accept()

    conversation_id = 0  # 0 asks the REST service to allocate a real PostgreSQL ID
    failure_count = 0

    try:
        while True:
            raw = await ws.receive_text()

            try:
                data = json.loads(raw)
                user_message = data.get("message", "")
                user_id = data.get("user_id", "user_001")
                conversation_id = data.get("conversation_id", conversation_id)
            except json.JSONDecodeError:
                user_message = raw
                user_id = "user_001"

            if not user_message.strip():
                continue

            try:
                response = await chat(ChatRequest(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message=user_message,
                ))
            except Exception:
                failure_count += 1
                if failure_count >= 3:
                    await ws.send_json({
                        "type": "handoff",
                        "reason": "agent_failure",
                        "message": "抱歉，我暂时无法处理，正在为您转接人工客服...",
                    })
                    await ws.close()
                    return
                await ws.send_json({
                    "type": "reply",
                    "content": "抱歉，出了点问题，请稍后重试。",
                    "done": True,
                })
                continue

            failure_count = 0  # reset on success
            conversation_id = response.conversation_id

            await ws.send_json({
                "type": "intent",
                "intent": response.intent,
                "conversation_id": conversation_id,
            })

            # Simulate streaming by sending character by character or chunked
            full_reply = response.reply
            chunk_size = 10
            for i in range(0, len(full_reply), chunk_size):
                chunk = full_reply[i : i + chunk_size]
                await ws.send_json({"type": "chunk", "content": chunk})
                await asyncio.sleep(0.02)  # simulate streaming delay

            await ws.send_json({
                "type": "reply",
                "content": full_reply,
                "conversation_id": conversation_id,
                "message_id": response.message_id,
                "trace": response.trace,
                "handoff": response.handoff,
                "handoff_reason": response.handoff_reason,
                "pending_confirmation": response.pending_confirmation,
                "done": True,
            })

    except WebSocketDisconnect:
        pass  # client disconnected
