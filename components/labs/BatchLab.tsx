import React, { useState, useEffect } from 'react';
import { Layers, Database, ArrowRight, CheckCircle, Clock } from 'lucide-react';

interface BatchItem {
  id: number;
  symptom: string;
  status: 'pending' | 'processing' | 'done';
  result?: string;
}

const SAMPLE_DATA: BatchItem[] = [
  { id: 1, symptom: '头晕目眩，耳鸣，口苦...', status: 'pending' },
  { id: 2, symptom: '面色苍白，神疲乏力...', status: 'pending' },
  { id: 3, symptom: '腰膝酸软，骨蒸潮热...', status: 'pending' },
  { id: 4, symptom: '胃脘胀痛，嗳气吞酸...', status: 'pending' },
  { id: 5, symptom: '咳嗽痰黄，胸闷...', status: 'pending' },
];

export const BatchLab: React.FC = () => {
  const [items, setItems] = useState<BatchItem[]>(SAMPLE_DATA);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const startBatch = () => {
    if (isRunning) return;
    setIsRunning(true);
    setItems(SAMPLE_DATA.map(i => ({ ...i, status: 'pending', result: undefined })));
    setProgress(0);
  };

  useEffect(() => {
    if (!isRunning) return;

    let currentIndex = 0;

    const processNext = () => {
      if (currentIndex >= items.length) {
        setIsRunning(false);
        return;
      }

      // Set current to processing
      setItems(prev => prev.map((item, idx) => 
        idx === currentIndex ? { ...item, status: 'processing' } : item
      ));

      // Simulate API latency + Sleep(0.2)
      setTimeout(() => {
        setItems(prev => prev.map((item, idx) => 
            idx === currentIndex ? { ...item, status: 'done', result: 'Success' } : item
        ));
        
        setProgress(((currentIndex + 1) / items.length) * 100);
        currentIndex++;
        processNext();
      }, 800); // Faster than real life for demo
    };

    processNext();
  }, [isRunning]);

  return (
    <div className="flex flex-col h-full bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
            <div className="flex items-center space-x-2 text-slate-700">
            <Layers className="w-5 h-5 text-orange-600" />
            <span className="font-semibold text-sm">Step 2: 批量处理逻辑 (Batch Processing)</span>
            </div>
            <div className="text-xs font-mono bg-orange-100 text-orange-700 px-2 py-1 rounded">
            for symptom in symptoms_data:
            </div>
        </div>

        <div className="flex-1 p-6 flex flex-col">
            <div className="mb-6 flex items-center justify-between">
                <div className="flex items-center space-x-4">
                    <Database className="w-10 h-10 text-slate-300" />
                    <div>
                        <h3 className="font-bold text-slate-700">Dataset Processing</h3>
                        <p className="text-xs text-slate-500">train_data.csv ({items.length} records)</p>
                    </div>
                </div>
                <button 
                    onClick={startBatch}
                    disabled={isRunning}
                    className={`px-4 py-2 rounded-lg text-sm font-bold shadow-sm transition-all ${
                        isRunning 
                        ? 'bg-slate-100 text-slate-400 cursor-wait' 
                        : 'bg-orange-500 text-white hover:bg-orange-600 active:scale-95'
                    }`}
                >
                    {isRunning ? 'Processing...' : 'Start Batch Job'}
                </button>
            </div>

            {/* Progress Bar */}
            <div className="w-full bg-slate-100 rounded-full h-4 mb-6 overflow-hidden">
                <div 
                    className="bg-orange-500 h-full transition-all duration-300 ease-out flex items-center justify-end px-2"
                    style={{ width: `${progress}%` }}
                >
                    <span className="text-[9px] text-white font-bold">{progress.toFixed(0)}%</span>
                </div>
            </div>

            {/* Visualization of the Loop */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
                {items.map((item) => (
                    <div 
                        key={item.id}
                        className={`p-3 rounded-lg border flex items-center justify-between transition-all duration-300 ${
                            item.status === 'processing' 
                                ? 'border-orange-400 bg-orange-50 scale-[1.02] shadow-md' 
                                : item.status === 'done' 
                                    ? 'border-green-200 bg-green-50/50 opacity-75' 
                                    : 'border-slate-100 bg-slate-50 opacity-50'
                        }`}
                    >
                        <div className="flex items-center space-x-3 overflow-hidden">
                            <span className="text-xs font-mono font-bold text-slate-400 w-6">#{item.id}</span>
                            <div className="truncate text-sm text-slate-700 w-48 md:w-64">{item.symptom}</div>
                        </div>

                        <div className="flex items-center space-x-3">
                            {item.status === 'processing' && (
                                <div className="flex items-center text-xs text-orange-600 font-bold animate-pulse">
                                    <Clock className="w-3 h-3 mr-1" />
                                    Inferencing...
                                </div>
                            )}
                            {item.status === 'done' && (
                                <div className="flex items-center text-xs text-green-600 font-bold">
                                    <CheckCircle className="w-3 h-3 mr-1" />
                                    Done
                                </div>
                            )}
                            {item.status === 'pending' && (
                                <div className="w-3 h-3 rounded-full bg-slate-200"></div>
                            )}
                        </div>
                    </div>
                ))}
            </div>

             {/* Principle Explanation */}
            <div className="mt-6 bg-orange-50 p-4 rounded-lg border border-orange-200 text-sm text-orange-900">
                <h4 className="font-bold mb-1 flex items-center"><Clock className="w-3 h-3 mr-2"/> Rate Limiting (速率限制)</h4>
                <p className="opacity-90">
                    代码中 <code>time.sleep(0.2)</code> 至关重要。API 提供商通常限制每秒请求数 (QPS)。如果没有这个暂停，循环速度会超过 API 处理能力，导致 <code>429 Too Many Requests</code> 错误。
                </p>
            </div>
        </div>
    </div>
  );
};
