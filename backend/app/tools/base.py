"""工具元数据结构：ToolDefinition / ToolParameterDef / ToolResult。"""

from __future__ import annotations  # 注解延迟求值：本模块只声明类型，不在类定义期求值 str | None

from dataclasses import dataclass, field  # field 只为可变默认值（list）存在，直接写 = [] 会在导入期 TypeError

from app.config import settings  # 只读超时/重试两个开关，本文件不做任何 IO


@dataclass
class ToolParameterDef:
    # 一个"模型可填写"的参数声明；校验（validate_arguments）和 schema 生成（to_openai_tools）都只读它，不看 handler 签名
    param_name: str  # 同时是 schema properties 的 key、normalized 的 key、handler 的 kwargs 名，三处必须一致
    param_type: str  # 必须是 JSON Schema 类型名；不在 registry._TYPE_COERCE 表里的值不报错，原样透传不转换
    param_desc: str = ""  # 给模型看的说明；缺必填时的报错文案「缺少必填参数 x（desc）」也直接复用它
    is_required: bool = False  # True 才进 schema 的 required；同时决定参数缺失时是报错还是回落 default_value
    default_value: str | None = None  # 类型只有 str：注入 normalized 时绕过转换表，所以 integer/boolean 的默认值靠 handler 自己兜


@dataclass
class ToolDefinition:
    name: str  # function calling 的机器名：模型回传的 tool_call.name 就是它，也是执行器 get() 唯一的查找依据
    display_name: str  # 只给后台/前端展示，不进注册键也不进 schema
    description: str  # 进 schema，是模型选不选这个工具的主要依据；写窄了漏调、写宽了乱调
    parameters: list[ToolParameterDef] = field(default_factory=list)  # 空列表＝无参数工具（CRM、售中的查询都是），schema 里 properties 为空对象
    version: str = "1.0"  # 只进注册键 name:version；执行器 get(name) 固定查 1.0，只升版本不保留 1.0 会直接"工具未注册"
    # 默认值取配置，让 TOOL_TIMEOUT_MS / TOOL_RETRY_COUNT 这两个开关真正生效
    # dataclass 默认值在 import 时求值一次，所以 TOOL_* 必须在应用加载本模块前就配好
    timeout_ms: int = settings.tool_timeout_ms  # 默认 5000ms（TOOL_TIMEOUT_MS）；web_search 显式覆写成 10000
    retry_count: int = settings.tool_retry_count  # 默认 3，是"额外重试次数"，最坏 4 次尝试；handler 自己吞掉的异常不会触发它
    # 由运行时注入的可信参数（如 user_id）。不写进给模型的 function schema，
    # 模型既看不到也无法伪造，越权查询在这一层就被挡住。
    context_params: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    # handler 的统一返回类型；执行器把它拆成普通 dict 才交给模型，工具函数不要直接返回 dict
    success: bool  # 业务级成败：False 也走"正常返回"分支，不抛错、不计入熔断失败
    data: object = None  # 原样进结果字典再拼进生成提示词（那边做 json.dumps），所以必须保持可 JSON 序列化
    error: str | None = None  # 面向排障的中文原因；当前生成节点只拼 data，error 未必被模型看到
    duration_ms: int = 0  # handler 自报耗时；invoke_tool 用的是自己的 monotonic 差值，这个字段目前不会出现在执行结果里
