import { useEffect, useState } from 'react';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import Input from '../components/ui/Input';
import Select from '../components/ui/Select';
import { fetchAgents, type AgentRow } from '../api/admin';

const DISPLAY: Record<string, string> = {
  pre_sales: '售前客服', during_sales: '售中客服',
  after_sales: '售后客服', complaint: '投诉受理',
};

export default function AgentConfig() {
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [activeName, setActiveName] = useState('after_sales');
  const [model, setModel] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [topK, setTopK] = useState('5');
  const [threshold, setThreshold] = useState('0.75');
  const [bm25Weight, setBm25Weight] = useState('0.3');

  useEffect(() => {
    fetchAgents().then(data => {
      setAgents(data.agents);
      const a = data.agents.find(a => a.name === 'after_sales');
      if (a) loadAgent(a);
    }).catch(() => {});
  }, []);

  const loadAgent = (a: AgentRow) => {
    setActiveName(a.name);
    setModel(a.model);
    setSystemPrompt(a.system_prompt || '');
    setTopK(a.retrieval_params?.top_k || '5');
    setThreshold(a.retrieval_params?.similarity_threshold || '0.75');
    setBm25Weight(a.retrieval_params?.bm25_weight || '0.3');
  };

  const switchAgent = (name: string) => {
    const a = agents.find(x => x.name === name);
    if (a) loadAgent(a);
  };

  const current = agents.find(a => a.name === activeName);
  const modelOptions = [{ value: model, label: model }];

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>Agent 运行配置</h1>
        <p className="text-xs mt-2" style={{ color: 'var(--text-secondary)' }}>
          展示当前进程真实生效的模型、Prompt 与检索参数；配置由 .env 和代码加载，修改后需重启后端。
        </p>
      </div>

      <div className="flex gap-3">
        {['pre_sales', 'during_sales', 'after_sales', 'complaint'].map(a => (
          <button key={a} onClick={() => switchAgent(a)}
                  className="px-4 py-2 text-sm font-medium transition-all"
                  style={{
                    color: activeName === a ? 'var(--accent-primary)' : 'var(--text-secondary)',
                    borderBottom: activeName === a ? '2px solid var(--accent-primary)' : '2px solid transparent',
                  }}>
            {DISPLAY[a]}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-base font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>System Prompt 模板</h3>
          <textarea
            value={systemPrompt}
            readOnly
            className="w-full h-48 px-3 py-2 text-sm rounded resize-none font-mono"
            style={{ background: 'var(--bg-tertiary)', border: '1px solid var(--border-default)', color: 'var(--text-primary)' }}
          />
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <h3 className="text-base font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>检索参数</h3>
            <div className="grid grid-cols-3 gap-4">
              <Input label="Top K" value={topK} readOnly />
              <Input label="相似度阈值" value={threshold} readOnly />
              <Input label="BM25 权重" value={bm25Weight} readOnly />
            </div>
          </Card>

          <Card>
            <h3 className="text-base font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>模型配置</h3>
            <Select label="生成模型" options={modelOptions} value={model} disabled />
            <div className="flex items-center gap-2 mt-3">
              <Badge variant="accent">状态</Badge>
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                {current?.status === 'active' ? '运行中' : current?.status ?? '-'} — {DISPLAY[activeName] || activeName}
              </span>
            </div>
          </Card>
        </div>
      </div>

      <div className="flex justify-end">
        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          配置来源：{current?.config_source === 'environment_and_code' ? '环境变量与 Agent Prompt 代码' : '后端运行时'}
        </span>
      </div>
    </div>
  );
}
