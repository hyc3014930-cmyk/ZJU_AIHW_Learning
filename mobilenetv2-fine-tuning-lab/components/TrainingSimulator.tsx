import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Play, RotateCcw } from 'lucide-react';
import { TrainingStrategy, ChartDataPoint } from '../types';
import { MOCK_HEAD_ONLY_DATA, MOCK_TWO_STEP_DATA } from '../constants';

const TrainingSimulator: React.FC = () => {
  const [strategy, setStrategy] = useState<TrainingStrategy>(TrainingStrategy.HEAD_ONLY);
  const [data, setData] = useState<ChartDataPoint[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);

  const startSimulation = () => {
    setIsSimulating(true);
    setData([]);
    
    const targetData = strategy === TrainingStrategy.HEAD_ONLY ? MOCK_HEAD_ONLY_DATA : MOCK_TWO_STEP_DATA;
    let currentStep = 0;

    const interval = setInterval(() => {
      if (currentStep >= targetData.length) {
        clearInterval(interval);
        setIsSimulating(false);
        return;
      }
      setData(prev => [...prev, targetData[currentStep]]);
      currentStep++;
    }, 100); // Speed of simulation
  };

  const reset = () => {
    setData([]);
    setIsSimulating(false);
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-800">训练策略模拟器</h2>
          <p className="text-sm text-slate-500">对比不同 Fine-tuning 策略下的准确率曲线</p>
        </div>
        
        <div className="flex gap-2">
            <select 
                value={strategy}
                onChange={(e) => {
                    setStrategy(e.target.value as TrainingStrategy);
                    reset();
                }}
                disabled={isSimulating}
                className="bg-slate-50 border border-slate-300 text-slate-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2.5"
            >
                <option value={TrainingStrategy.HEAD_ONLY}>策略一：仅训 Head</option>
                <option value={TrainingStrategy.TWO_STEP}>策略二：两步法 (Head + 整网)</option>
            </select>
            
            <button
                onClick={startSimulation}
                disabled={isSimulating || data.length > 0}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isSimulating || data.length > 0 
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                    : 'bg-indigo-600 text-white hover:bg-indigo-700'
                }`}
            >
                <Play className="w-4 h-4" /> 开始训练
            </button>
            <button
                onClick={reset}
                className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
                title="重置"
            >
                <RotateCcw className="w-5 h-5" />
            </button>
        </div>
      </div>

      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="epoch" label={{ value: 'Epoch (轮数)', position: 'insideBottomRight', offset: -5 }} stroke="#64748b" />
            <YAxis domain={[0, 100]} label={{ value: '准确率 (%)', angle: -90, position: 'insideLeft' }} stroke="#64748b" />
            <Tooltip 
                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                labelStyle={{ color: '#64748b', marginBottom: '4px' }}
            />
            <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#4f46e5" 
                strokeWidth={3} 
                dot={{ r: 3, fill: '#4f46e5' }} 
                activeDot={{ r: 6 }} 
                animationDuration={300}
            />
            {strategy === TrainingStrategy.TWO_STEP && (
                <ReferenceLine x={10} stroke="#f97316" strokeDasharray="3 3" label={{ position: 'top', value: 'Unfreeze', fill: '#f97316', fontSize: 12 }} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={`p-4 rounded-lg border ${strategy === TrainingStrategy.HEAD_ONLY ? 'bg-indigo-50 border-indigo-200' : 'bg-slate-50 border-slate-200'}`}>
            <h4 className="font-bold text-sm mb-1">策略一：仅训 Head</h4>
            <p className="text-xs text-slate-600">
                训练极快，曲线迅速上升但较早进入瓶颈（约 85%）。适合快速验证或简单任务。
            </p>
        </div>
        <div className={`p-4 rounded-lg border ${strategy === TrainingStrategy.TWO_STEP ? 'bg-indigo-50 border-indigo-200' : 'bg-slate-50 border-slate-200'}`}>
            <h4 className="font-bold text-sm mb-1">策略二：两步法</h4>
            <p className="text-xs text-slate-600">
                前10轮冻结Backbone快速收敛。第11轮解冻（Unfreeze），利用通用特征+专属特征，突破瓶颈达到更高精度（约 96%）。
            </p>
        </div>
      </div>
    </div>
  );
};

export default TrainingSimulator;