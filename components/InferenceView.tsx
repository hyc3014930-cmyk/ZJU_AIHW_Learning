import React, { useState } from 'react';
import { Search, UserCheck, UserX, ArrowRight, FileCode, CheckCircle, Smartphone, Play, Loader2, MousePointerClick, AlignCenterHorizontal, Braces } from 'lucide-react';

export const InferenceView: React.FC = () => {
  const [processingNode, setProcessingNode] = useState<number | null>(null);
  const [results, setResults] = useState<{[key: number]: boolean}>({});

  const handlePredict = (nodeId: number) => {
    setProcessingNode(nodeId);
    
    setTimeout(() => {
      setProcessingNode(null);
      setResults(prev => ({ ...prev, [nodeId]: true }));
    }, 1500);
  };

  return (
    <div className="h-full flex flex-col animate-fade-in-up">
      <div className="p-6 border-b border-slate-100 bg-slate-50/50">
        <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          <Search className="text-amber-600" /> 第四步：预测推理 (Inference)
        </h2>
        <p className="text-slate-600 mt-2 text-sm">
           模型训练好后，我们就可以用它来抓坏人了。这里的关键是将概率转换为具体的分类标签。
        </p>
      </div>

      <div className="flex-1 bg-slate-50/30 overflow-hidden flex flex-col lg:flex-row">
        
        {/* Left: Interaction Area */}
        <div className="flex-1 p-6 overflow-y-auto">
            
            <div className="flex flex-col gap-6 max-w-2xl mx-auto">
                {/* Visual Pipeline */}
                <div className="bg-white p-8 rounded-2xl border border-slate-200 shadow-sm flex items-center justify-between relative overflow-hidden">
                     {processingNode !== null && (
                         <div className="absolute top-0 bottom-0 left-0 w-full bg-gradient-to-r from-transparent via-amber-100/50 to-transparent animate-[shimmer_1.5s_infinite] pointer-events-none"></div>
                     )}
                     <style>{`@keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }`}</style>
                     
                     <div className="text-center z-10 flex flex-col items-center">
                        <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-xl mb-2 transition-colors ${processingNode !== null ? 'bg-amber-100 text-amber-600' : 'bg-slate-100 text-slate-400'}`}>
                           <Smartphone />
                        </div>
                        <div className="text-[10px] font-bold text-slate-500 uppercase">Input Data</div>
                     </div>
                     
                     <ArrowRight className="text-slate-300" />
                     
                     <div className="text-center z-10 flex flex-col items-center">
                        <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-xl mb-2 transition-all ${processingNode !== null ? 'bg-blue-600 text-white shadow-lg scale-110' : 'bg-slate-100 text-slate-400'}`}>
                           <FileCode />
                        </div>
                        <div className="text-[10px] font-bold text-slate-500 uppercase">MLP Model</div>
                     </div>

                     <ArrowRight className="text-slate-300" />
                     
                     {/* Argmax Logic */}
                     <div className="text-center z-10 flex flex-col items-center">
                        <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-xl mb-2 transition-colors ${!processingNode && Object.keys(results).length > 0 ? 'bg-purple-100 text-purple-600' : 'bg-slate-100 text-slate-400'}`}>
                           <AlignCenterHorizontal />
                        </div>
                        <div className="text-[10px] font-bold text-slate-500 uppercase">Argmax</div>
                     </div>

                     <ArrowRight className="text-slate-300" />

                     <div className="text-center z-10 flex flex-col items-center">
                         <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-xl mb-2 transition-colors ${!processingNode && Object.keys(results).length > 0 ? 'bg-emerald-100 text-emerald-600' : 'bg-slate-100 text-slate-400'}`}>
                           <CheckCircle />
                        </div>
                        <div className="text-[10px] font-bold text-slate-500 uppercase">Final Label</div>
                     </div>
                </div>

                {/* Cards */}
                <h3 className="font-bold text-slate-700 flex items-center gap-2 mt-4">
                    <MousePointerClick size={18} />
                    选择用户进行测试
                </h3>
                
                {[0, 1].map(id => (
                    <div key={id} className="bg-white rounded-xl border border-slate-200 p-4 flex items-center gap-4 hover:shadow-md transition-shadow">
                        <div className="bg-slate-100 w-12 h-12 rounded-lg flex items-center justify-center font-mono font-bold text-slate-600">
                            {id}
                        </div>
                        <div className="flex-1">
                            <div className="text-sm font-bold text-slate-800">User ID: {id}</div>
                            <div className="text-xs text-slate-500">Node feature vector ready</div>
                        </div>
                        
                        {results[id] ? (
                            <div className={`flex flex-col items-end animate-fade-in-up`}>
                                <div className="text-xs text-slate-400 mb-1 font-mono">
                                   argmax({id === 0 ? '[0.98, 0.02]' : '[0.15, 0.85]'})
                                </div>
                                <div className={`px-3 py-1 rounded-full text-xs font-bold border ${id === 0 ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-rose-50 text-rose-700 border-rose-200'}`}>
                                    {id === 0 ? 'Label 0: 正常用户' : 'Label 1: 欺诈用户'}
                                </div>
                            </div>
                        ) : (
                            <button 
                                onClick={() => handlePredict(id)}
                                disabled={processingNode !== null}
                                className="bg-blue-600 text-white px-4 py-2 rounded-lg text-xs font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                                {processingNode === id ? <Loader2 className="animate-spin" size={14}/> : 'Predict'}
                            </button>
                        )}
                    </div>
                ))}
            </div>
        </div>

        {/* Right: Code */}
        <div className="w-full lg:w-[450px] bg-slate-900 border-l border-slate-700 p-6 shadow-xl overflow-y-auto">
             <div className="flex items-center gap-2 text-slate-400 mb-6 pb-2 border-b border-slate-700">
                <FileCode size={16} />
                <span className="text-sm font-bold">predict.py</span>
             </div>

             <div className="font-mono text-xs leading-loose space-y-4">
                 
                 <div className={`p-3 rounded transition-all ${processingNode !== null ? 'bg-blue-900/30 border border-blue-500/30' : 'opacity-50'}`}>
                     <code className="block text-yellow-200 mb-1">def predict(data, node_id):</code>
                     <code className="block text-slate-400 pl-4"># 1. 开启“只读模式”，不计算梯度</code>
                     <code className="block text-purple-300 pl-4">with torch.no_grad():</code>
                     <code className="block text-purple-300 pl-8">model.eval()</code>
                     <code className="block text-blue-300 pl-8">out = model(data.x[node_id])</code>
                 </div>

                 <div className={`p-3 rounded transition-all ${!processingNode && Object.keys(results).length > 0 ? 'bg-emerald-900/30 border border-emerald-500/30' : 'opacity-50'}`}>
                     <code className="block text-slate-400 pl-4"># 2. 将 Log 概率转为真实概率</code>
                     <code className="block text-emerald-300 pl-4">y_pred = out.exp()</code>
                     <code className="block text-yellow-200 pl-4">return y_pred</code>
                 </div>

                 <div className={`p-3 rounded transition-all ${!processingNode && Object.keys(results).length > 0 ? 'bg-purple-900/30 border border-purple-500/30' : 'opacity-50'}`}>
                     <code className="block text-slate-400"># 3. 找出概率最大的那个类 (核心逻辑)</code>
                     <code className="block text-slate-300">y_pred = predict(data, node_idx)</code>
                     <code className="block text-purple-300 font-bold">label_index = torch.argmax(y_pred)</code>
                     
                     <code className="block text-slate-400 mt-2"># 4. 翻译给人类看</code>
                     <code className="block text-blue-300">dic = {'{'}0: "正常用户", 1: "欺诈用户"{'}'}</code>
                     <code className="block text-emerald-300">print(f"标签为: {'{'}dic[label_index.item()]{'}'}")</code>
                 </div>

             </div>

             <div className="mt-8 bg-slate-800 p-4 rounded-xl border border-slate-700 text-xs text-slate-400">
                 <h4 className="font-bold text-slate-200 mb-2 flex items-center gap-2">
                    <Braces size={14} className="text-purple-400"/>
                    为什么要用 Argmax?
                 </h4>
                 <p className="leading-relaxed mb-2">
                     <code>torch.argmax</code> 会返回数组中<strong>最大值所在的索引</strong>。
                 </p>
                 <div className="bg-slate-900 p-2 rounded font-mono text-xs mb-2">
                    输入: [0.1, 0.9] <br/>
                    最大值: 0.9 <br/>
                    索引: 1 (代表欺诈)
                 </div>
                 <p className="leading-relaxed">
                     虽然概率是 [0.1, 0.9]，但我们不能说用户是“10%的正常人”。最终分类必须是非黑即白的，所以选概率最大的那个作为最终结果。
                 </p>
             </div>
        </div>

      </div>
    </div>
  );
};