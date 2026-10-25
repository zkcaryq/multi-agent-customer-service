# 基于 Multi-Agent 的智能客服业务平台

多 Agent 协作 + RAG 混合检索 + Function Calling 的企业级电商智能客服系统，覆盖**售前、售中、售后、投诉**全链路；高危业务操作走人工审批，附带离线评测与运营后台。

## 核心特性

- **四领域 Agent 协同**：Supervisor 选择主 Agent；复合诉求并行调用相关领域 Agent 做只读咨询，由主 Agent 汇总并负责业务操作
- **混合检索 RAG**：Milvus 向量 + BM25 关键词（0.7/0.3 加权融合），bge-reranker 重排序，235 个知识切片
- **多格式知识库**：Markdown（文档结构）· FAQ（Q&A边界）· PDF/Word（语义切片）· Excel（按行切表）· 纯文本（滑动窗口）
- **LangChain Function Calling**：模型与工具协议由 LangChain 适配，17 个业务工具由统一执行器完成校验、调用和留痕
- **Human-in-the-loop**：退款、修改地址、创建投诉工单等高风险操作暂停等待审批；用户端与运营后台都能批准/拒绝，超时未处理由数据库侧自动判为拒绝；PostgreSQL Checkpointer 支持跨进程恢复执行
- **运营后台闭环**：人工审批队列、人工工单回填、审计日志（审批/退款/建单/登录留痕）、反馈看板、链路监控；后台所有接口走 JWT + 角色校验
- **工具容错**：指数退避重试（1s→2s→4s）+ 熔断器（连续 5 次失败自动熔断 30s）+ 网络/业务错误区分
- **联网搜索降级**：DuckDuckGo，知识库无匹配时自动查询网页
- **LLM 记忆压缩**：摘要模型按岗位配置（`LLM_MODEL_SUMMARY`），超 8 轮自动压缩，Redis 持久化 24h TTL
- **转人工兜底**：用户主动 / 工具全失败 / 知识缺失三种触发 + WebSocket 3 次连续失败兜底
- **敏感信息脱敏**：身份证/银行卡/手机号/邮箱自动检测打码，安全检测前先脱敏
- **实时流式**：WebSocket 流式输出 + HTTP 降级，打字机光标 + 反馈追问
- **链路监控**：管理端展示检索/重排/生成耗时、模型 Token、工具成功率、调用明细和待审批数量

## 架构

```
用户消息 → 内容安全 + 脱敏 → Qwen 1.5B 意图分类 → 路由 Agent
  → LangGraph: 检索 → 重排序 → 多轮工具规划 → 高风险审批 → 工具调用(重试+熔断) → 生成
  → LLM 摘要压缩 → Redis 持久化 → WebSocket 流式 / Handoff 转人工
```

## 技术栈

| 层面 | 选型 |
|------|------|
| 后端框架 | Python FastAPI + LangChain 模型/工具适配 + LangGraph 对话编排 |
| LLM | Qwen Max / Plus / 1.5B（DashScope API） |
| Embedding | text-embedding-v3（1024 维） |
| 向量库 | Milvus Lite（开发）/ Milvus Server（生产） |
| 关键词检索 | BM25Okapi（内存 + pickle 持久化） |
| 重排序 | bge-reranker-v2-m3 / 分数降级 |
| 缓存 / 记忆 | Redis（会话持久化，24h TTL） |
| 数据库 | PostgreSQL + SQLAlchemy 2.0 async |
| 前端 | React + TypeScript + WebSocket + Tailwind CSS |

### 业务系统适配边界

订单、CRM、退款和工单工具遵循统一 Tool 契约；当前演示环境使用项目内 PostgreSQL
作为本地适配实现，包含真实读写、用户归属校验和事务处理。生产接入外部系统时只需
替换工具 handler 为 HTTP / MCP 客户端，Agent、参数校验、审批和审计链路保持不变。

## 快速开始

```bash
# 1. 配置环境变量
cp backend/.env.example backend/.env   # 填模型网关/Key、数据库、Redis、向量库等

# 2. 安装依赖（在仓库根目录；pyproject.toml 是唯一依赖真源）
uv sync

# 3. 注入知识库（首次；之后启动会自动加载 BM25 + Milvus 索引）
cd backend && uv run python scripts/ingest.py --rebuild

# 4. 造演示订单（跑订单/退款/审批相关自检前需要）
uv run python scripts/seed_orders.py

# 5. 启动后端（建表、引导后台账号都在启动时自动完成）
uv run python run.py                    # Windows 必须走这个入口（详见 run.py 顶部说明）
# Linux/macOS 也可以：uv run uvicorn app.main:app --reload --port 8000

# 6. 启动前端（两个都要起，默认端口都是 5173，所以第二个要指定端口）
cd ../frontend/admin  && npm install && npm run dev                 # 管理端 http://localhost:5173
cd ../frontend/widget && npm install && npm run dev -- --port 5175  # 用户端 http://localhost:5175
```

管理端需要登录，账号由 `ADMIN_BOOTSTRAP_USER` / `ADMIN_BOOTSTRAP_PASSWORD` 引导（默认 `admin`），
首次创建后改口令请走数据库，重启不会覆盖。

## 项目结构

```
backend/
├── app/
│   ├── agents/
│   │   ├── base_agent.py        # 领域 Agent 基类：检索→重排→决策→审批→执行→生成
│   │   ├── coordinator.py       # Supervisor 协同：主 Agent + 并行只读咨询
│   │   ├── router.py            # LLM 意图分类 + 路由分发
│   │   └── pre_sales/ during_sales/ after_sales/ complaint/
│   ├── providers/               # chat / embedding / rerank 供应商抽象与装配
│   ├── rag/                     # 分切器、向量化、Milvus、BM25、混合检索、重排、注入
│   ├── core/
│   │   ├── auth.py              # JWT + pbkdf2 口令 + 角色校验
│   │   ├── approvals.py         # 审批队列：登记、超时自动拒绝、结论回写
│   │   ├── hitl.py              # 高危工具表 + 审批语义解析 + 转人工判定
│   │   ├── checkpointer.py      # LangGraph PostgreSQL 检查点（interrupt/resume 落盘）
│   │   ├── audit.py             # 敏感动作留痕
│   │   ├── log.py               # 级别/格式可配 + trace_id
│   │   └── memory.py security.py resilience.py timeutil.py
│   ├── tools/                   # registry（校验/注入上下文）、executor（超时/重试/熔断）、crm
│   ├── evaluation/              # 离线评测指标（意图准确率、检索 hit/MRR、转人工判定）
│   ├── api/                     # chat / websocket / admin / auth
│   └── models/                  # database.py（33 表）+ engine.py（async SQLAlchemy）
├── scripts/                     # 注入、造数、9 个自检脚本、离线评测
├── tests/                       # pytest：单元测试 + 集成自检（无 PG 自动跳过）
├── run.py                       # Windows 推荐入口（Selector 事件循环，见文件头）
└── data/                        # 知识库文档（9 文件 / 235 chunks）

frontend/
├── admin/                       # 运营后台：仪表盘 / 知识库 / 工具 / Agent 配置 /
│                                #   对话记录 / 人工审批 / 人工工单 / 反馈 / 审计 / 链路监控
└── widget/                      # 用户端：流式对话 + 反馈 + 转人工 + 高危操作审批面板
```

接口自检与回归：

```bash
uv run pytest -m "not integration"   # 单元测试（不需要外部服务）
uv run pytest                        # 全量：单元 + 9 个端到端自检（需要 PG/Redis，约 5 分钟）
```

离线评测（都要真实模型，跑完只打印不落库）：

```bash
uv run python scripts/run_evaluation.py     # 单问单答：意图准确率 / 检索命中 / MRR / 转人工判定
uv run python scripts/run_complex_eval.py   # 复杂任务：多轮对话 + 工具 + 审批 + 数据库终态
```

「复杂任务成功率」的口径：`data/eval/complex_tasks.json` 里每条用例声明断言
（必须调用的工具、关键参数、是否进人工审批、审批结论、落库终态、回答必须包含的事实），
**全部满足才算通过**，通过数 / 总数即成功率。判定全部是规则断言，不引入 LLM 当裁判，
所以结果可复现、失败能定位到具体断言。每条用例使用专属 `ORD-EVAL-*` 订单，
跑前自动重置、跑完自动删除（这些订单会进用户画像统计，留着会污染演示数据）。
单次约 3-4 分钟；`EVAL_ONLY=1,3` 只跑指定用例，`EVAL_DEBUG=1` 打印每条的回复与工具调用。


## MVP 完成度

- [x] Widget 前端 + WebSocket 流式 + 暗黑主题
- [x] LLM 意图分类 + 关键词降级 + 四领域 Agent 路由
- [x] Markdown / FAQ / PDF / Word / Excel 多格式知识库
- [x] Milvus + BM25 混合检索 + bge-reranker 重排序
- [x] 17 个 Function Calling 工具 + 参数校验、超时重试、熔断容错
- [x] 退款/修改地址/投诉工单 Human-in-the-loop 审批与断点恢复
- [x] LLM 对话摘要 + Redis 持久化记忆
- [x] Agent 内 Handoff 转人工 + WebSocket 失败兜底
- [x] 敏感信息自动脱敏 + 内容安全检测
- [x] 反馈按钮 + 追问弹窗 + PostgreSQL 全量持久化
- [x] Admin 后台（仪表盘/对话回放/反馈看板/知识库管理/Agent配置/工具管理/文档上传）
- [x] 链路监控（Trace耗时/Token/工具调用/审批状态）
- [x] 联网搜索降级（知识库无匹配时自动查网页）
- [x] 100 款商品知识库（235 个切片，11 个品类）
