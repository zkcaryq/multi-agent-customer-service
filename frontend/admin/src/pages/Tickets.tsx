import { useCallback, useEffect, useState } from 'react';
import { CheckCircle2 } from 'lucide-react';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import type { Variant } from '../components/ui/Badge';
import Button from '../components/ui/Button';
import Table from '../components/ui/Table';
import Select from '../components/ui/Select';
import Modal from '../components/ui/Modal';
import { showToast } from '../components/ui/Toast';
import { fetchTickets, resolveTicket, type TicketRow } from '../api/admin';

const STATUS: Record<string, { label: string; variant: Variant }> = {
  pending:   { label: '待处理', variant: 'warning' },
  resolved:  { label: '已解决', variant: 'success' },
  closed:    { label: '已关闭', variant: 'default' },
};

const FILTERS = [
  { value: 'pending', label: '待处理' },
  { value: 'resolved', label: '已解决' },
  { value: '', label: '全部' },
];

function fmt(ts: string | null): string {
  return ts ? ts.replace('T', ' ').slice(0, 19) : '-';
}

export default function Tickets() {
  const [status, setStatus] = useState('pending');
  const [rows, setRows] = useState<TicketRow[]>([]);
  const [target, setTarget] = useState<TicketRow | null>(null);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    fetchTickets(status)
      .then((r) => setRows(r.items))
      .catch((e: Error) => showToast('error', e.message));
  }, [status]);

  useEffect(() => { load(); }, [load]);

  const submit = async () => {
    if (!target) return;
    setBusy(true);
    try {
      await resolveTicket(target.id, note);
      showToast('success', `工单 ${target.ticket_id} 已回填处理结论`);
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
    { key: 'ticket_id', header: '工单号', width: '110px', render: (r: TicketRow) => (
      <span className="font-mono text-sm">{r.ticket_id}</span>
    )},
    { key: 'category', header: '类型', width: '100px', render: (r: TicketRow) => <Badge variant="accent">{r.category}</Badge> },
    { key: 'summary', header: '摘要', render: (r: TicketRow) => (
      <span className="text-sm">{r.summary || '-'}</span>
    )},
    { key: 'from_agent', header: '来源', width: '110px', render: (r: TicketRow) => (
      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{r.from_agent}</span>
    )},
    { key: 'customer', header: '客户', width: '110px', render: (r: TicketRow) => (
      <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>{r.customer}</span>
    )},
    { key: 'status', header: '状态', width: '90px', render: (r: TicketRow) => {
      const s = STATUS[r.status] || { label: r.status, variant: 'default' as Variant };
      return <Badge variant={s.variant}>{s.label}</Badge>;
    }},
    { key: 'created_at', header: '创建', width: '150px', render: (r: TicketRow) => (
      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{fmt(r.created_at)}</span>
    )},
    { key: 'resolution', header: '处理结论', render: (r: TicketRow) => (
      r.resolution_note
        ? <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{r.resolution_note}</span>
        : <span className="text-xs" style={{ color: 'var(--text-disabled)' }}>-</span>
    )},
    { key: 'ops', header: '', width: '60px', render: (r: TicketRow) => (
      r.status === 'pending' ? (
        <button title="回填处理结论" onClick={() => { setTarget(r); setNote(''); }}
                className="p-1.5 rounded hover:bg-[rgba(78,205,196,0.12)]" style={{ color: 'var(--accent-success)' }}>
          <CheckCircle2 size={15} />
        </button>
      ) : null
    )},
  ];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>人工工单</h1>
          <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
            投诉 Agent 建单、Agent 判定转人工的会话都会落到这里
          </p>
        </div>
        <div style={{ width: '160px' }}>
          <Select options={FILTERS} value={status} onChange={(e) => setStatus(e.target.value)} />
        </div>
      </div>

      <Card>
        <Table columns={columns} data={rows} emptyText="暂无工单" />
      </Card>

      <Modal open={!!target} onClose={() => setTarget(null)} title={`处理工单 ${target?.ticket_id ?? ''}`}>
        {target && (
          <div className="flex flex-col gap-4">
            <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              <div>客户 {target.customer} · 会话 <span className="font-mono">#{target.conversation_id}</span></div>
              <div className="mt-1">类型：{target.category}</div>
              <div className="mt-1">摘要：{target.summary || '-'}</div>
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>处理结论</label>
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                rows={3}
                placeholder="例如：已电话联系客户并补发，客户确认收到"
                className="px-3 py-2 text-sm rounded resize-none"
                style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-default)', color: 'var(--text-primary)' }}
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" size="sm" onClick={() => setTarget(null)}>取消</Button>
              <Button size="sm" disabled={busy} onClick={submit}>{busy ? '提交中…' : '标记已解决'}</Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
