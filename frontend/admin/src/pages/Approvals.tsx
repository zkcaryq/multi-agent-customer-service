import { useCallback, useEffect, useState } from 'react';
import { Check, X } from 'lucide-react';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import Button from '../components/ui/Button';
import Table from '../components/ui/Table';
import Select from '../components/ui/Select';
import Modal from '../components/ui/Modal';
import { showToast } from '../components/ui/Toast';
import {
  decideApproval,
  fetchApprovals,
  type ApprovalRow,
} from '../api/admin';
import type { Variant } from '../components/ui/Badge';

const STATUS: Record<string, { label: string; variant: Variant }> = {
  pending:  { label: '待审批', variant: 'warning' },
  approved: { label: '已批准', variant: 'success' },
  rejected: { label: '已拒绝', variant: 'danger' },
  expired:  { label: '超时拒绝', variant: 'default' },
};

const FILTERS = [
  { value: 'pending', label: '待审批' },
  { value: 'approved', label: '已批准' },
  { value: 'rejected', label: '已拒绝' },
  { value: 'expired', label: '超时拒绝' },
  { value: '', label: '全部' },
];

function fmt(ts: string | null): string {
  return ts ? ts.replace('T', ' ').slice(0, 19) : '-';
}

export default function Approvals() {
  const [status, setStatus] = useState('pending');
  const [rows, setRows] = useState<ApprovalRow[]>([]);
  const [target, setTarget] = useState<{ row: ApprovalRow; approved: boolean } | null>(null);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    fetchApprovals(status)
      .then((r) => setRows(r.items))
      .catch((e: Error) => showToast('error', e.message));
  }, [status]);

  useEffect(() => { load(); }, [load]);

  const submit = async () => {
    if (!target) return;
    setBusy(true);
    try {
      const res = await decideApproval(target.row.id, target.approved, note);
      // 超时后批准会被系统降级为拒绝，这里必须按后端真正生效的结论提示
      if (res.handoff_reason === 'approval_rejected') {
        showToast('warning', '审批窗口已过，系统已自动拒绝并转人工');
      } else {
        showToast('success', target.approved ? '已批准并恢复执行' : '已拒绝，会话转人工');
      }
      setTarget(null);
      setNote('');
      load();
    } catch (e) {
      showToast('warning', (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const columns = [
    { key: 'id', header: 'ID', width: '60px', render: (r: ApprovalRow) => <span className="font-mono text-sm">#{r.id}</span> },
    { key: 'agent', header: 'Agent', width: '110px', render: (r: ApprovalRow) => <Badge variant="accent">{r.agent}</Badge> },
    { key: 'conv', header: '会话 / 用户', width: '130px', render: (r: ApprovalRow) => (
      <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>#{r.conversation_id} · {r.user_id}</span>
    )},
    { key: 'actions', header: '待执行动作', render: (r: ApprovalRow) => (
      <div className="flex flex-col gap-1">
        {r.actions.map((a, i) => (
          <div key={i} className="text-xs">
            <span className="font-medium" style={{ color: 'var(--accent-warning)' }}>{a.label || a.tool}</span>
            <code className="ml-2 font-mono" style={{ color: 'var(--text-secondary)' }}>
              {JSON.stringify(a.arguments)}
            </code>
          </div>
        ))}
      </div>
    )},
    { key: 'requested_at', header: '请求时间', width: '160px', render: (r: ApprovalRow) => (
      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{fmt(r.requested_at)}</span>
    )},
    { key: 'status', header: '状态', width: '100px', render: (r: ApprovalRow) => {
      const s = STATUS[r.status] || { label: r.status, variant: 'default' as Variant };
      return <Badge variant={s.variant}>{s.label}</Badge>;
    }},
    { key: 'decision', header: '审批人 / 备注', width: '180px', render: (r: ApprovalRow) => (
      r.decided_by ? (
        <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          <div>{r.decided_by}</div>
          {r.decision_note && <div className="truncate" title={r.decision_note}>{r.decision_note}</div>}
        </div>
      ) : <span className="text-xs" style={{ color: 'var(--text-disabled)' }}>等待处理</span>
    )},
    { key: 'ops', header: '', width: '120px', render: (r: ApprovalRow) => (
      r.status === 'pending' ? (
        <div className="flex gap-1">
          <button title="批准" onClick={() => { setTarget({ row: r, approved: true }); setNote(''); }}
                  className="p-1.5 rounded hover:bg-[rgba(78,205,196,0.12)]" style={{ color: 'var(--accent-success)' }}>
            <Check size={15} />
          </button>
          <button title="拒绝" onClick={() => { setTarget({ row: r, approved: false }); setNote(''); }}
                  className="p-1.5 rounded hover:bg-[rgba(224,85,106,0.12)]" style={{ color: 'var(--accent-danger)' }}>
            <X size={15} />
          </button>
        </div>
      ) : null
    )},
  ];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>人工审批</h1>
          <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
            高危工具（退款、改地址、建投诉单）在这里人工放行；超时未处理的会被系统自动拒绝
          </p>
        </div>
        <div style={{ width: '160px' }}>
          <Select options={FILTERS} value={status} onChange={(e) => setStatus(e.target.value)} />
        </div>
      </div>

      <Card>
        <Table columns={columns} data={rows} emptyText="没有符合条件的审批请求" />
      </Card>

      <Modal
        open={!!target}
        onClose={() => setTarget(null)}
        title={target?.approved ? '批准高危操作' : '拒绝高危操作'}
      >
        {target && (
          <div className="flex flex-col gap-4">
            <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              <div>会话 <span className="font-mono">#{target.row.conversation_id}</span> · 用户 {target.row.user_id}</div>
              <div className="mt-1">
                动作：
                {target.row.actions.map((a) => `${a.label || a.tool} ${JSON.stringify(a.arguments)}`).join('；')}
              </div>
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
                审批备注（会写进审计日志）
              </label>
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                rows={3}
                placeholder={target.approved ? '例如：已核对订单金额与用户身份' : '例如：金额异常，需人工复核'}
                className="px-3 py-2 text-sm rounded resize-none"
                style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-default)', color: 'var(--text-primary)' }}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" size="sm" onClick={() => setTarget(null)}>取消</Button>
              <Button
                variant={target.approved ? 'primary' : 'danger'}
                size="sm"
                disabled={busy}
                onClick={submit}
              >
                {busy ? '提交中…' : target.approved ? '确认批准并执行' : '确认拒绝'}
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
