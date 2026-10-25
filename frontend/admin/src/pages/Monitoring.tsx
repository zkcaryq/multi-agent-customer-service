import { useCallback, useEffect, useState } from 'react';
import { Activity, Clock, Database, RefreshCw, ShieldAlert, Wrench } from 'lucide-react';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import {
  fetchMonitorSummary,
  fetchMonitorTraces,
  type MonitorSummary,
  type TraceRow,
} from '../api/admin';

export default function Monitoring() {
  const [summary, setSummary] = useState<MonitorSummary | null>(null);
  const [traces, setTraces] = useState<TraceRow[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [summaryData, traceData] = await Promise.all([
        fetchMonitorSummary(24),
        fetchMonitorTraces(50),
      ]);
      setSummary(summaryData);
      setTraces(traceData.items);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load().catch(() => {}); }, [load]);

  const cards = [
    { label: '24小时链路', value: summary?.trace_count ?? '-', icon: Activity },
    { label: '平均生成耗时', value: summary ? `${summary.avg_generation_ms} ms` : '-', icon: Clock },
    { label: '工具成功率', value: summary?.tool_success_rate == null ? '暂无调用' : `${summary.tool_success_rate}%`, icon: Wrench },
    { label: '待审批', value: summary?.pending_approvals ?? '-', icon: ShieldAlert },
  ];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>链路监控</h1>
          <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
            消息级 Trace、检索耗时、模型 Token、工具调用和审批状态
          </p>
        </div>
        <button onClick={() => load()} disabled={loading}
          className="flex items-center gap-2 px-3 py-2 text-sm rounded disabled:opacity-50"
          style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border-default)' }}>
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />刷新
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {cards.map((item) => (
          <Card key={item.label} accent>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{item.label}</div>
                <div className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>{item.value}</div>
              </div>
              <item.icon size={32} style={{ color: 'var(--border-default)' }} />
            </div>
          </Card>
        ))}
      </div>

      <Card>
        <div className="flex flex-wrap gap-6 text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          <span>平均检索：<b style={{ color: 'var(--text-primary)' }}>{summary?.avg_retrieval_ms ?? '-'} ms</b></span>
          <span>工具调用：<b style={{ color: 'var(--text-primary)' }}>{summary?.tool_calls ?? '-'}</b></span>
          <span>失败：<b style={{ color: 'var(--accent-danger)' }}>{summary?.tool_failures ?? '-'}</b></span>
          <span>Token：<b style={{ color: 'var(--text-primary)' }}>{summary ? summary.prompt_tokens + summary.completion_tokens : '-'}</b></span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ color: 'var(--text-secondary)', borderBottom: '1px solid var(--border-default)' }}>
                {['Trace', 'Agent / 模型', '检索', '生成', 'Token', '工具', '时间'].map((label) => (
                  <th key={label} className="text-left px-2 py-2 font-medium">{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {traces.map((trace) => (
                <tr key={trace.trace_id} style={{ borderBottom: '1px solid var(--border-default)' }}>
                  <td className="px-2 py-3 font-mono">#{trace.trace_id}<div className="text-xs" style={{ color: 'var(--text-disabled)' }}>会话 {trace.conversation_id}</div></td>
                  <td className="px-2 py-3"><Badge variant="default">{trace.agent || 'unknown'}</Badge><div className="text-xs mt-1">{trace.model || '-'}</div></td>
                  <td className="px-2 py-3">{trace.retrieval_count ?? 0} 条<div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{trace.retrieval_ms ?? 0} ms + 重排 {trace.rerank_ms ?? 0} ms</div></td>
                  <td className="px-2 py-3 font-mono">{trace.generation_ms ?? 0} ms</td>
                  <td className="px-2 py-3 font-mono">{(trace.prompt_tokens ?? 0) + (trace.completion_tokens ?? 0)}</td>
                  <td className="px-2 py-3">
                    {trace.tool_calls.length === 0 ? <span style={{ color: 'var(--text-disabled)' }}>无</span> : trace.tool_calls.map((call, index) => (
                      <div key={`${call.name}-${index}`} className="flex items-center gap-1 mb-1">
                        <span style={{ color: call.success ? 'var(--accent-success)' : 'var(--accent-danger)' }}>{call.success ? '●' : '×'}</span>
                        <span>{call.name}</span><span className="text-xs">{call.duration_ms ?? 0}ms</span>
                      </div>
                    ))}
                  </td>
                  <td className="px-2 py-3 text-xs whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>{trace.created_at?.replace('T', ' ').slice(0, 19) || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!traces.length && <div className="py-10 text-center text-sm" style={{ color: 'var(--text-disabled)' }}><Database size={28} className="mx-auto mb-2" />暂无链路数据</div>}
        </div>
      </Card>
    </div>
  );
}
