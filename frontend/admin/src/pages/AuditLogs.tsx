import { useCallback, useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import type { Variant } from '../components/ui/Badge';
import Table from '../components/ui/Table';
import Input from '../components/ui/Input';
import { showToast } from '../components/ui/Toast';
import { fetchAuditLogs, type AuditRow } from '../api/admin';

const QUICK_ACTIONS = [
  { value: '', label: '全部' },
  { value: 'approval_decided', label: '审批' },
  { value: 'refund_submitted', label: '退款' },
  { value: 'complaint_ticket_created', label: '投诉建单' },
  { value: 'ticket_resolved', label: '工单处理' },
  { value: 'admin_login', label: '后台登录' },
];

const ACTION_VARIANT: Record<string, Variant> = {
  approval_decided: 'accent',
  refund_submitted: 'success',
  complaint_ticket_created: 'warning',
  ticket_resolved: 'success',
  admin_login: 'default',
};

function fmt(ts: string | null): string {
  return ts ? ts.replace('T', ' ').slice(0, 19) : '-';
}

export default function AuditLogs() {
  const [action, setAction] = useState('');
  const [keyword, setKeyword] = useState('');
  const [rows, setRows] = useState<AuditRow[]>([]);

  const load = useCallback(() => {
    fetchAuditLogs(action)
      .then((r) => setRows(r.items))
      .catch((e: Error) => showToast('error', e.message));
  }, [action]);

  useEffect(() => { load(); }, [load]);

  const visible = keyword
    ? rows.filter((r) => (r.summary + r.action + (r.detail ?? '')).toLowerCase().includes(keyword.toLowerCase()))
    : rows;

  const columns = [
    { key: 'created_at', header: '时间', width: '160px', render: (r: AuditRow) => (
      <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>{fmt(r.created_at)}</span>
    )},
    { key: 'action', header: '动作', width: '190px', render: (r: AuditRow) => (
      <Badge variant={ACTION_VARIANT[r.action] || 'default'}>{r.action}</Badge>
    )},
    { key: 'entity', header: '对象', width: '150px', render: (r: AuditRow) => (
      <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>{r.entity_type}#{r.entity_id}</span>
    )},
    { key: 'summary', header: '摘要', render: (r: AuditRow) => (
      <span className="text-sm" title={r.detail || undefined}>{r.summary}</span>
    )},
    { key: 'actor', header: '操作人', width: '110px', render: (r: AuditRow) => (
      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
        {r.admin_user_id ? `staff#${r.admin_user_id}` : 'system'}
      </span>
    )},
  ];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>审计日志</h1>
          <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
            审批结论、退款受理、投诉建单、工单处理、后台登录都留痕，按动作或关键字过滤
          </p>
        </div>
        <button onClick={load} className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded hover:bg-[var(--bg-tertiary)]"
                style={{ color: 'var(--text-secondary)' }}>
          <RefreshCw size={13} /> 刷新
        </button>
      </div>

      <div className="flex items-end gap-3 flex-wrap">
        <div className="flex gap-1.5 flex-wrap">
          {QUICK_ACTIONS.map((a) => (
            <button
              key={a.value}
              onClick={() => setAction(a.value)}
              className="px-3 py-1.5 text-xs rounded transition-colors"
              style={{
                background: action === a.value ? 'rgba(0,212,228,0.12)' : 'var(--bg-tertiary)',
                color: action === a.value ? 'var(--accent-primary)' : 'var(--text-secondary)',
                border: `1px solid ${action === a.value ? 'var(--border-accent)' : 'var(--border-default)'}`,
              }}
            >
              {a.label}
            </button>
          ))}
        </div>
        <div className="ml-auto" style={{ width: '220px' }}>
          <Input placeholder="按关键字过滤当前结果…" value={keyword} onChange={(e) => setKeyword(e.target.value)} />
        </div>
      </div>

      <Card>
        <Table columns={columns} data={visible} emptyText="暂无审计记录" />
      </Card>
    </div>
  );
}
