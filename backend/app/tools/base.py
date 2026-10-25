from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings


@dataclass
class ToolParameterDef:
    param_name: str
    param_type: str
    param_desc: str = ""
    is_required: bool = False
    default_value: str | None = None


@dataclass
class ToolDefinition:
    name: str
    display_name: str
    description: str
    parameters: list[ToolParameterDef] = field(default_factory=list)
    version: str = "1.0"
    # 默认值取配置，让 TOOL_TIMEOUT_MS / TOOL_RETRY_COUNT 这两个开关真正生效
    timeout_ms: int = settings.tool_timeout_ms
    retry_count: int = settings.tool_retry_count
    # 由运行时注入的可信参数（如 user_id）。不写进给模型的 function schema，
    # 模型既看不到也无法伪造，越权查询在这一层就被挡住。
    context_params: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    success: bool
    data: object = None
    error: str | None = None
    duration_ms: int = 0
