import React, { useState } from 'react';
import { Database, RefreshCw, Play, FileCode, Info } from 'lucide-react';

export const DataView: React.FC = () => {
  const [isNormalized, setIsNormalized] = useState(false);

  const dataPoints = [
    { id: 1, raw: 80, norm: 10, label: 'x1' },
    { id: 2, raw: 20, norm: -5, label: 'x2' },
    { id: 3, raw: 95, norm: 15, label: 'x3' },
    { id: 4, raw: 10, norm: -12, label: 'x4' },
    { id: 5, raw: 60, norm: 2, label: 'x5' },
    { id: 6, raw: 30, norm: -8, label: 'x6' },
  ];

  return (
    <div className="h-full flex flex-col animate-fade-in-up">
      {/* Header */}
      <div className="p-6 border-b border-slate-100 bg-slate-50/50">
        <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          <Database className="text-blue-600" /> 第一步：数据准备 (Data Preparation)
        </h2>
        <p className="text-slate-600 mt-2 text-sm">
          任何 AI 模型的第一步都是“喂数据”。这里展示了如何加载 DGraphFin 数据集，并进行关键的“归一化”处理。
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 bg-slate-50/30">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column: Visualization */}
          <div className="lg:col-span-2 space-y-8">
            
            {/* 1. Loading */}
            <section className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm relative overflow-hidden">
               <div className="absolute top-0 right-0 p-4 opacity-10">
                  <Database size={100} />
               </div>
               <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                  <span className="bg-blue-100 text-blue-700 w-6 h-6 rounded-full flex items-center justify-center text-xs">1</span>
                  加载 DGraphFin 数据集
               </h3>
               <div className="grid grid-cols-2 gap-4 relative z-10">
                  <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 hover:shadow-md transition-shadow">
                     <div className="text-xs text-slate-500 font-bold uppercase mb-1">Features (特征矩阵)</div>
                     <div className="font-mono text-lg font-bold text-slate-700">data.x</div>
                     <div className="text-xs text-slate-400 mt-1">Shape: [N, 17] (N个用户, 17个特征)</div>
                  </div>
                  <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 hover:shadow-md transition-shadow">
                     <div className="text-xs text-slate-500 font-bold uppercase mb-1">Labels (标签向量)</div>
                     <div className="font-mono text-lg font-bold text-slate-700">data.y</div>
                     <div className="text-xs text-slate-400 mt-1">Shape: [N] (每个用户对应 0 或 1)</div>
                  </div>
               </div>
            </section>

            {/* 2. Normalization */}
            <section className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
               <div className="flex justify-between items-center mb-6">
                  <h3 className="font-bold text-slate-800 flex items-center gap-2">
                     <span className="bg-purple-100 text-purple-700 w-6 h-6 rounded-full flex items-center justify-center text-xs">2</span>
                     归一化处理 (Normalization)
                  </h3>
                  <button 
                    onClick={() => setIsNormalized(!isNormalized)}
                    className="flex items-center gap-2 text-xs font-bold bg-purple-600 text-white px-3 py-1.5 rounded-lg shadow-sm hover:bg-purple-700 transition-colors"
                  >
                    {isNormalized ? <RefreshCw size={12}/> : <Play size={12}/>}
                    {isNormalized ? '重置数据 (Reset)' : '执行归一化 (Run)'}
                  </button>
               </div>

               <div className="bg-slate-50 rounded-xl border border-slate-200 h-48 flex items-center justify-center relative overflow-hidden group">
                   {/* Visualizing Data Points */}
                   <div className="absolute w-full h-px bg-slate-300 top-1/2 left-0 z-0"></div>
                   {isNormalized && (
                     <div className="absolute right-2 top-1/2 -translate-y-6 text-[10px] font-bold text-purple-600 bg-purple-100 px-2 py-0.5 rounded animate-fade-in-up">
                       Mean ≈ 0, Std = 1
                     </div>
                   )}
                   
                   <div className="flex items-end gap-6 h-32 z-10">
                      {dataPoints.map((point) => {
                        return (
                          <div key={point.id} className="flex flex-col items-center gap-2 relative w-6">
                             <div 
                                className={`w-3 rounded-full transition-all duration-1000 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${isNormalized ? 'bg-purple-500' : 'bg-slate-400'}`}
                                style={{
                                  height: `${isNormalized ? Math.abs(point.norm * 3) + 4 : point.raw}px`,
                                  transform: `translateY(${isNormalized ? (point.norm > 0 ? '-1px' : '1px') : '64px'})`,
                                  opacity: 0.8
                                }}
                             ></div>
                             <span className="text-[10px] text-slate-400 font-mono absolute -bottom-8">
                                {isNormalized ? point.norm.toFixed(0) : point.raw}
                             </span>
                          </div>
                        );
                      })}
                   </div>
               </div>
               <p className="text-xs text-slate-500 mt-4 bg-yellow-50 p-2 rounded border border-yellow-100 flex gap-2">
                  <Info size={14} className="text-yellow-600 flex-shrink-0 mt-0.5" />
                  <span>
                     <strong>原理：</strong> 这行代码 <code>(x - mean) / std</code> 叫做 Z-Score 标准化。它把原来“参差不齐”的数据（有的几万，有的只有几）强行拉到了同一个起跑线上（均值为0，方差为1），防止模型因为数值大小差异而产生偏见。
                  </span>
               </p>
            </section>
            
            {/* 3. Split */}
            <section className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                  <span className="bg-emerald-100 text-emerald-700 w-6 h-6 rounded-full flex items-center justify-center text-xs">3</span>
                  划分数据集 (Train / Valid / Test)
               </h3>
               <div className="flex w-full h-10 rounded-lg overflow-hidden font-bold text-[10px] text-white text-center shadow-inner leading-10">
                  <div className="bg-emerald-500 w-[70%] hover:bg-emerald-600 transition-colors cursor-help border-r border-emerald-600/20" title="train_mask">
                    train_mask (训练用)
                  </div>
                  <div className="bg-amber-500 w-[15%] hover:bg-amber-600 transition-colors cursor-help border-r border-amber-600/20" title="valid_mask">
                    valid (验证用)
                  </div>
                  <div className="bg-slate-400 w-[15%] hover:bg-slate-500 transition-colors cursor-help" title="test_mask">
                    test (考试用)
                  </div>
               </div>
            </section>

          </div>

          {/* Right Column: Code Explanation */}
          <div className="lg:col-span-1 space-y-6">
            
            <div className="bg-slate-800 rounded-xl overflow-hidden shadow-lg border border-slate-700 flex flex-col h-full max-h-[600px]">
              <div className="bg-slate-900 px-4 py-3 flex items-center gap-2 border-b border-slate-700">
                 <FileCode size={14} className="text-blue-400" />
                 <span className="text-xs font-bold text-slate-300">data_process.py (源代码)</span>
              </div>
              <div className="p-4 overflow-y-auto font-mono text-xs leading-relaxed custom-scrollbar">
                
                <code className="block text-slate-500 mb-2"># 导入必要的库</code>
                <code className="block text-pink-300">from utils import DGraphFin</code>
                <code className="block text-pink-300">import torch_geometric.transforms as T</code>
                <code className="block text-pink-300 mb-4">import torch</code>

                <code className="block text-slate-500"># 1. 设置设备 (GPU/CPU)</code>
                <code className="block text-blue-300">device = f'cuda:0' if torch.cuda.is_available() else 'cpu'</code>
                <code className="block text-blue-300 mb-4">device = torch.device(device)</code>

                <code className="block text-slate-500"># 2. 加载数据</code>
                <code className="block text-slate-300">path = './datasets/...'</code>
                <code className="block text-slate-300">dataset_name = 'DGraph'</code>
                <code className="block text-blue-300">dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())</code>
                <code className="block text-slate-300">nlabels = dataset.num_classes <span className="text-slate-500"># = 2</span></code>
                <code className="block text-slate-300 mb-4">data = dataset[0]</code>

                <code className="block text-slate-500"># 3. 归一化 (核心代码)</code>
                <code className="block text-slate-500"># 将特征值转化为标准正态分布</code>
                <code className="block text-purple-300">x = data.x</code>
                <div className={`transition-colors duration-500 ${isNormalized ? 'bg-purple-900/50 -mx-4 px-4 border-l-2 border-purple-400' : ''}`}>
                    <code className="block text-purple-300 font-bold">x = (x - x.mean(0)) / x.std(0)</code>
                </div>
                <code className="block text-purple-300 mb-4">data.x = x</code>

                <code className="block text-slate-500"># 4. 准备训练索引</code>
                <code className="block text-emerald-300">split_idx = {'{'}</code>
                <code className="block text-emerald-300 pl-4">'train': data.train_mask,</code>
                <code className="block text-emerald-300 pl-4">'valid': data.valid_mask,</code>
                <code className="block text-emerald-300 pl-4">'test':  data.test_mask</code>
                <code className="block text-emerald-300">{'}'}</code>
              </div>
            </div>

          </div>

        </div>
      </div>
    </div>
  );
};