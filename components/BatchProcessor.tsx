import React, { useEffect, useRef, useState } from 'react';

type Item = { id: number; text: string };

const BatchProcessor: React.FC<{ items: Item[]; onComplete?: () => void }> = ({ items, onComplete }) => {
  const [status, setStatus] = useState<'idle' | 'running' | 'finished'>('idle');
  const [active, setActive] = useState<number | null>(null);
  const [processed, setProcessed] = useState<number[]>([]);
  const timers = useRef<number[]>([]);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      timers.current.forEach(t => clearTimeout(t));
      timers.current = [];
    };
  }, []);

  const start = () => {
    if (status === 'running') return;
    setStatus('running');
    setProcessed([]);
    setActive(null);

    let idx = 0;

    const step = () => {
      if (!mounted.current) return;
      if (idx >= items.length) {
        setStatus('finished');
        setActive(null);
        if (onComplete) onComplete();
        return;
      }

      setActive(idx);
      // simulate API delay
      const t1 = window.setTimeout(() => {
        if (!mounted.current) return;
        setProcessed(prev => [...prev, idx]);
        idx += 1;
        // small gap
        const t2 = window.setTimeout(() => {
          step();
        }, 200);
        timers.current.push(t2);
      }, 400);
      timers.current.push(t1);
    };

    step();
  };

  const reset = () => {
    timers.current.forEach(t => clearTimeout(t));
    timers.current = [];
    setStatus('idle');
    setActive(null);
    setProcessed([]);
  };

  return (
    <div className="bg-white p-4 rounded-xl border border-slate-200">
      <div className="flex items-center justify-between mb-3">
        <div className="font-bold text-slate-700">批量处理演示</div>
        <div className="flex gap-2">
          <button onClick={start} className="px-3 py-1 bg-emerald-600 text-white rounded text-xs">Start</button>
          <button onClick={reset} className="px-3 py-1 bg-slate-100 rounded text-xs">Reset</button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2">
        {items.map((it, i) => (
          <div key={it.id} className={`p-2 rounded flex items-center justify-between border ${processed.includes(i) ? 'bg-emerald-50 border-emerald-200' : active===i ? 'bg-yellow-50 border-yellow-200' : 'bg-white border-slate-100'}`}>
            <div>
              <div className="text-sm font-mono text-slate-700">{it.text}</div>
              <div className="text-xs text-slate-400">ID: {it.id}</div>
            </div>
            <div className="text-xs text-slate-500">
              {processed.includes(i) ? '已处理' : active===i ? '处理中...' : '等待'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BatchProcessor;
