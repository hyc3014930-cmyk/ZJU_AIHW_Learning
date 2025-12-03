import React, { useState } from 'react';
import { BookOpen, Share2, ShieldAlert, ShieldCheck, Zap, Users } from 'lucide-react';

export const IntroView: React.FC = () => {
  const [modelType, setModelType] = useState<'MLP' | 'GNN'>('MLP');

  return (
    <div className="h-full flex flex-col animate-fade-in-up">
      <div className="p-6 border-b border-slate-100 bg-slate-50/50">
        <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
          <BookOpen className="text-slate-700" /> 0. 项目背景 (Context)
        </h2>
        <p className="text-slate-600 mt-2 text-sm">
          了解为什么我们要使用神经网络来检测金融欺诈，以及 DGraph 数据集是什么。
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 bg-slate-50/30">
        
        {/* Section 1: Business Context */}
        <section className="mb-10">
            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                <ShieldAlert className="text-rose-500"/>
                1. 金融反欺诈场景
            </h3>
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col md:flex-row gap-8 items-center">
                <div className="flex-1">
                    <p className="text-slate-600 text-sm leading-relaxed mb-4">
                        在互联网金融信贷业务中，<strong className="text-rose-600">欺诈用户</strong>可能会伪造身份申请贷款，然后逾期不还。
                        反欺诈技术的目标就是在大规模的用户中，通过分析他们的行为特征，精准地把这些“坏人”找出来。
                    </p>
                    <div className="flex gap-4">
                        <div className="flex items-center gap-2 text-xs bg-emerald-50 text-emerald-700 px-3 py-1.5 rounded-full border border-emerald-100">
                            <ShieldCheck size={14}/> 正常用户 (Label 0)
                        </div>
                        <div className="flex items-center gap-2 text-xs bg-rose-50 text-rose-700 px-3 py-1.5 rounded-full border border-rose-100">
                            <ShieldAlert size={14}/> 欺诈用户 (Label 1)
                        </div>
                    </div>
                </div>
                {/* Visual */}
                <div className="w-full md:w-1/3 h-40 relative bg-slate-50 rounded-xl border border-slate-100 overflow-hidden flex items-center justify-center">
                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 100">
                        {/* Edges */}
                        <line x1="50" y1="50" x2="100" y2="30" stroke="#cbd5e1" strokeWidth="1" />
                        <line x1="50" y1="50" x2="100" y2="70" stroke="#cbd5e1" strokeWidth="1" />
                        <line x1="150" y1="50" x2="100" y2="30" stroke="#cbd5e1" strokeWidth="1" />
                        <line x1="150" y1="50" x2="100" y2="70" stroke="#cbd5e1" strokeWidth="1" />
                        
                        {/* Nodes */}
                        <circle cx="50" cy="50" r="8" fill="#10b981" />
                        <circle cx="100" cy="30" r="8" fill="#10b981" />
                        <circle cx="100" cy="70" r="8" fill="#f43f5e" className="animate-pulse" /> {/* Fraud */}
                        <circle cx="150" cy="50" r="8" fill="#10b981" />
                    </svg>
                    <div className="absolute bottom-2 right-2 text-[10px] text-slate-400">关系网络示意图</div>
                </div>
            </div>
        </section>

        {/* Section 2: Dataset */}
        <section className="mb-10">
            <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                <DatabaseIcon />
                2. DGraphFin 数据集
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                    <h4 className="font-bold text-slate-700 text-sm mb-2">📊 数据规模</h4>
                    <ul className="text-xs text-slate-500 space-y-2">
                        <li className="flex justify-between border-b border-slate-50 pb-1">
                            <span>节点 (用户数):</span> <span className="font-mono text-slate-700">3,700,550</span>
                        </li>
                        <li className="flex justify-between border-b border-slate-50 pb-1">
                            <span>边 (关系数):</span> <span className="font-mono text-slate-700">4,300,999</span>
                        </li>
                         <li className="flex justify-between border-b border-slate-50 pb-1">
                            <span>特征维度 (x):</span> <span className="font-mono text-slate-700">17</span>
                        </li>
                         <li className="flex justify-between">
                            <span>欺诈比例:</span> <span className="font-mono text-slate-700">~1.3% (严重不平衡)</span>
                        </li>
                    </ul>
                </div>
                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                    <h4 className="font-bold text-slate-700 text-sm mb-2">⏱️ 动态图特性</h4>
                    <p className="text-xs text-slate-500 leading-relaxed">
                        DGraph 是由真实金融场景中随着时间演变事件构成的。这意味着用户之间的关系网是动态变化的，虽然本项目为了简化，暂时将其作为静态图处理（使用 MLP 进行节点分类）。
                    </p>
                </div>
            </div>
        </section>

        {/* Section 3: Model Comparison */}
        <section>
             <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                <Zap className="text-amber-500"/>
                3. 技术路线：MLP vs GNN
            </h3>
            
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="flex border-b border-slate-100">
                    <button 
                        onClick={() => setModelType('MLP')}
                        className={`flex-1 py-3 text-sm font-bold transition-colors ${modelType === 'MLP' ? 'bg-blue-50 text-blue-600 border-b-2 border-blue-600' : 'text-slate-500 hover:bg-slate-50'}`}
                    >
                        本项目：MLP (多层感知机)
                    </button>
                    <button 
                        onClick={() => setModelType('GNN')}
                        className={`flex-1 py-3 text-sm font-bold transition-colors ${modelType === 'GNN' ? 'bg-purple-50 text-purple-600 border-b-2 border-purple-600' : 'text-slate-500 hover:bg-slate-50'}`}
                    >
                        拓展：GNN (图神经网络)
                    </button>
                </div>
                
                <div className="p-8 flex flex-col items-center">
                    <div className="relative w-64 h-64 mb-6">
                         {/* Common Graph Background */}
                         <svg className="absolute inset-0 w-full h-full" viewBox="0 0 200 200">
                             {/* Neighbors */}
                            <line x1="100" y1="100" x2="50" y2="50" stroke="#e2e8f0" strokeWidth="2" strokeDasharray="4"/>
                            <line x1="100" y1="100" x2="150" y2="50" stroke="#e2e8f0" strokeWidth="2" strokeDasharray="4"/>
                            <line x1="100" y1="100" x2="100" y2="160" stroke="#e2e8f0" strokeWidth="2" strokeDasharray="4"/>
                            
                            <circle cx="50" cy="50" r="15" fill="white" stroke="#94a3b8" strokeWidth="2"/>
                            <circle cx="150" cy="50" r="15" fill="white" stroke="#94a3b8" strokeWidth="2"/>
                            <circle cx="100" cy="160" r="15" fill="white" stroke="#94a3b8" strokeWidth="2"/>
                            
                            {/* Target Node */}
                            <circle cx="100" cy="100" r="20" fill={modelType === 'MLP' ? '#3b82f6' : '#9333ea'} className="shadow-lg"/>
                            <text x="100" y="105" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">User</text>

                            {/* Animations */}
                            {modelType === 'MLP' ? (
                                <>
                                    {/* MLP: Isolation */}
                                    <circle cx="100" cy="100" r="25" fill="none" stroke="#60a5fa" strokeWidth="2" className="animate-ping" opacity="0.5"/>
                                </>
                            ) : (
                                <>
                                    {/* GNN: Aggregation */}
                                    <path d="M 50 50 L 100 100" stroke="#a855f7" strokeWidth="2" className="animate-draw"/>
                                    <path d="M 150 50 L 100 100" stroke="#a855f7" strokeWidth="2" className="animate-draw"/>
                                    <path d="M 100 160 L 100 100" stroke="#a855f7" strokeWidth="2" className="animate-draw"/>
                                    <circle cx="50" cy="50" r="15" fill="#e9d5ff" stroke="#a855f7" strokeWidth="2" className="animate-pulse"/>
                                    <circle cx="150" cy="50" r="15" fill="#e9d5ff" stroke="#a855f7" strokeWidth="2" className="animate-pulse"/>
                                    <circle cx="100" cy="160" r="15" fill="#e9d5ff" stroke="#a855f7" strokeWidth="2" className="animate-pulse"/>
                                </>
                            )}
                         </svg>
                    </div>

                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-sm max-w-lg w-full">
                        <div className="flex gap-2 mb-2 font-bold text-slate-700">
                             {modelType === 'MLP' ? <Users size={18} className="text-slate-400"/> : <Share2 size={18} className="text-purple-500"/>}
                             {modelType === 'MLP' ? '视角：孤立节点' : '视角：社交关系'}
                        </div>
                        <p className="text-slate-600 leading-relaxed text-xs">
                            {modelType === 'MLP' 
                                ? "MLP 只关注当前用户自身的特征（例如：年龄、存款、消费记录）。它把每个用户当成孤岛，虽然计算快，但忽略了“近朱者赤，近墨者黑”的社交关系。" 
                                : "GNN (图神经网络) 不仅看用户自己，还会聚合邻居的信息（例如：如果你经常和欺诈用户转账，那你也可能是欺诈用户）。这在反欺诈领域通常能提供更高的准确率。"}
                        </p>
                    </div>
                </div>
            </div>
        </section>

      </div>
    </div>
  );
};

// Helper Icon
const DatabaseIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
);
