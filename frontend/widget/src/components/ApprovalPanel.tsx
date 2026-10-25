import { useState } from 'react';
import type { PendingApproval } from '../types';

export default function ApprovalPanel({ approval, onDecide }: {
  approval: PendingApproval;
  onDecide: (approved: boolean, note?: string) => Promise<void>;
}) {
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const decide = async (approved: boolean) => {
    setSubmitting(true);
    try {
      await onDecide(approved, note);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-3 mb-3 p-3 ark-border" style={{ background: 'var(--bg-tertiary)' }}>
      <div className="text-sm font-bold mb-2" style={{ color: 'var(--accent-primary)' }}>
        人工审批
      </div>
      <div className="text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>
        {approval.prompt}
      </div>
      <ul className="text-sm mb-2 space-y-1" style={{ color: 'var(--text-primary)' }}>
        {approval.actions.map((action, index) => (
          <li key={`${action.tool}-${index}`}>• {action.label}</li>
        ))}
      </ul>
      <input
        value={note}
        onChange={(event) => setNote(event.target.value)}
        placeholder="审批备注（可选）"
        disabled={submitting}
        className="w-full px-2 py-1.5 mb-2 text-xs"
        style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)', border: '1px solid var(--border-default)' }}
      />
      <div className="flex gap-2">
        <button
          onClick={() => decide(true)} disabled={submitting}
          className="flex-1 px-3 py-1.5 text-xs font-bold disabled:opacity-50"
          style={{ background: 'var(--accent-success)', color: '#fff' }}
        >批准并执行</button>
        <button
          onClick={() => decide(false)} disabled={submitting}
          className="flex-1 px-3 py-1.5 text-xs font-bold disabled:opacity-50"
          style={{ background: 'var(--accent-danger)', color: '#fff' }}
        >拒绝操作</button>
      </div>
    </div>
  );
}
