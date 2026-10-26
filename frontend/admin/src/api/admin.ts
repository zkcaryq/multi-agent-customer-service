/** Admin panel API client — dashboard, conversations, feedback, knowledge, agents, tools. */

const BASE = 'http://localhost:8000/api/v1/admin';

export interface DashboardStats {
  total_conversations: number;
  today_conversations: number;
  active_conversations: number;
  total_messages: number;
  handoff_count: number;
  satisfaction: number | null;
  total_feedback: number;
  agent_distribution: { name: string; count: number }[];
}

export interface MonitorSummary {
  window_hours: number;
  trace_count: number;
  avg_generation_ms: number;
  avg_retrieval_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  tool_calls: number;
  tool_failures: number;
  tool_success_rate: number | null;
  pending_approvals: number;
}

export interface TraceRow {
  trace_id: number;
  conversation_id: number;
  message_id: number;
  agent: string | null;
  intent_confidence: number | null;
  model: string | null;
  retrieval_count: number | null;
  retrieval_ms: number | null;
  rerank_ms: number | null;
  generation_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  created_at: string | null;
  tool_calls: Array<{
    name: string;
    success: boolean;
    duration_ms: number | null;
    parameters: string | null;
    result: string | null;
  }>;
}

export interface ConvRow {
  id: number;
  user_id: number;
  agent_name: string;
  status: string;
  title: string;
  message_count: number;
  last_message_at: string | null;
  created_at: string | null;
}

export interface ConvMessages {
  conversation_id: number;
  messages: { id: number; role: string; content: string; created_at: string | null }[];
}

export interface FbStats {
  total: number;
  helpful: number;
  unhelpful: number;
  satisfaction_rate: number | null;
  pending_review: number;
}

export interface FbRow {
  id: number;
  message_id: number;
  rating: string;
  reason: string | null;
  review_status: string;
  created_at: string | null;
}

export interface DocRow {
  id: number;
  title: string;
  file_type: string;
  split_method: string;
  chunk_count: number;
  total_retrievals: number;
  status: string;
  version: number;
  created_at: string | null;
}

export interface KnowledgeStats {
  total_documents: number;
  total_chunks: number;
  total_retrievals: number;
}

export interface AgentRow {
  id: number;
  name: string;
  display_name: string;
  model: string;
  system_prompt: string | null;
  status: string;
  retrieval_params: Record<string, string>;
  config_source?: string;
}

export interface ToolRow {
  id: number;
  name: string;
  display_name: string;
  description: string;
  http_method: string;
  endpoint_url: string;
  version: string;
  status: string;
  timeout_ms: number;
  retry_count: number;
  agent_id?: number | null;
}

function authHeaders(extra?: HeadersInit): Record<string, string> {
  const token = localStorage.getItem('ma-cs-admin-token');
  return { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...(extra as object) };
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, { ...options, headers: authHeaders(options?.headers) });
  if (res.status === 401) {
    // 令牌过期/被撤销：回登录页，避免页面停在假数据上
    localStorage.removeItem('ma-cs-admin-token');
    if (!location.pathname.endsWith('/login')) location.href = '/login';
    throw new Error('登录已过期');
  }
  if (!res.ok) {
    const error = await res.json().catch(() => null);
    throw new Error(error?.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// Dashboard
export const fetchDashboard = () => fetchJson<DashboardStats>(`${BASE}/dashboard`);

export const fetchMonitorSummary = (hours = 24) =>
  fetchJson<MonitorSummary>(`${BASE}/monitor/summary?hours=${hours}`);

export const fetchMonitorTraces = (limit = 50) =>
  fetchJson<{ items: TraceRow[] }>(`${BASE}/monitor/traces?limit=${limit}`);

// Conversations
export const fetchConversations = (page = 1, status = '') =>
  fetchJson<{ rows: ConvRow[]; page: number }>(`${BASE}/conversations?page=${page}&page_size=50&status=${status}`);

export const fetchConversationMessages = (convId: number) =>
  fetchJson<ConvMessages>(`${BASE}/conversations/${convId}/messages`);

// Feedback
export const fetchFeedback = (page = 1, rating = '') =>
  fetchJson<{ rows: FbRow[]; page: number }>(`${BASE}/feedback?page=${page}&page_size=50&rating=${rating}`);

export const fetchFeedbackStats = () => fetchJson<FbStats>(`${BASE}/feedback/stats`);

// Knowledge
export const fetchKnowledge = (page = 1) =>
  fetchJson<{ rows: DocRow[]; page: number }>(`${BASE}/knowledge?page=${page}&page_size=50`);

export const fetchKnowledgeStats = () => fetchJson<KnowledgeStats>(`${BASE}/knowledge/stats`);

// Agents
export const fetchAgents = () => fetchJson<{ agents: AgentRow[] }>(`${BASE}/agents`);

export const updateAgent = (id: number, body: Record<string, unknown>) =>
  fetchJson<{ status: string }>(`${BASE}/agents/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

// Tools
export const fetchAdminTools = () => fetchJson<{ tools: ToolRow[] }>(`${BASE}/tools`);

// Approvals（HITL 人工审批队列）
export interface ApprovalAction {
  tool: string;
  label?: string;
  arguments: Record<string, unknown>;
}

export interface ApprovalRow {
  id: number;
  agent: string;
  conversation_id: number;
  user_id: string;
  actions: ApprovalAction[];
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  requested_at: string | null;
  decided_at: string | null;
  decided_by: string | null;
  decision_note: string | null;
  timeout_s: number;
}

export const fetchApprovals = (status = '') =>
  fetchJson<{ items: ApprovalRow[] }>(`${BASE}/approvals${status ? `?status=${status}` : ''}`);

export const decideApproval = (id: number, approved: boolean, note = '') =>
  fetchJson<{ reply?: string; handoff?: boolean; handoff_reason?: string }>(
    `${BASE}/approvals/${id}/decide`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ approved, note }) },
  );

// Tickets（人工工单）
export interface TicketRow {
  id: number;
  ticket_id: string;
  category: string;
  summary: string;
  from_agent: string;
  status: string;
  customer: string;
  conversation_id: number;
  created_at: string | null;
  resolved_at: string | null;
  resolution_note: string | null;
}

export const fetchTickets = (status = '') =>
  fetchJson<{ items: TicketRow[] }>(`${BASE}/tickets${status ? `?status=${status}` : ''}`);

export const resolveTicket = (id: number, note = '', status = 'resolved') =>
  fetchJson<{ ticket_status: string }>(`${BASE}/tickets/${id}/resolve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note, status }),
  });

// Audit（敏感动作留痕）
export interface AuditRow {
  id: number;
  action: string;
  entity_type: string;
  entity_id: number;
  summary: string;
  detail: string | null;
  admin_user_id: number | null;
  created_at: string | null;
}

export const fetchAuditLogs = (action = '', limit = 100) => {
  const qs = new URLSearchParams({ limit: String(limit) });
  if (action) qs.set('action', action);
  return fetchJson<{ items: AuditRow[] }>(`${BASE}/audit-logs?${qs}`);
};

// Upload
export interface UploadResult {
  status: string;
  filename: string;
  document_id: number;
  chunks: number;
  split_method: string;
  file_type: string;
  elapsed_ms: number;
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  return fetchJson<UploadResult>(`${BASE}/knowledge/upload`, {
    method: 'POST',
    body: form,
  });
}
