import React, { useState, useEffect } from 'react';
import { Ruler, RefreshCw, Calculator, ArrowRight } from 'lucide-react';
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine } from 'recharts';

export const EmbeddingLab: React.FC = () => {
  const [textA, setTextA] = useState("肝火上炎证");
  const [textB, setTextB] = useState("肝火上炎");
  const [similarity, setSimilarity] = useState(0);
  
  // Mock embeddings visualization data
  const [chartData, setChartData] = useState<any[]>([]);

  // Function to simulate embedding generation and similarity calculation
  const calculate = () => {
    // Basic text overlap similarity for demo (real world uses vector math)
    const setA = new Set(textA.split(''));
    const setB = new Set(textB.split(''));
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    const union = new Set([...setA, ...setB]);
    
    // Jaccard similarity as a proxy for semantic similarity demo
    let sim = intersection.size / union.size;
    
    // Boost score if identical or contained to simulate semantic match
    if (textA === textB) sim = 1.0;
    else if (textA.includes(textB) || textB.includes(textA)) sim = 0.8 + (sim * 0.2);
    
    // Random noise to simulate "embedding space" nuances
    sim = Math.min(0.99, Math.max(0.01, sim + (Math.random() * 0.1 - 0.05)));

    setSimilarity(sim);
    
    // Generate mock 2D points to represent high-dimensional vector projection
    const angleA = Math.random() * Math.PI * 0.5;
    const angleB = angleA + (1 - sim) * 1.5; // More similar = closer angle
    
    setChartData([
        { x: Math.cos(angleA) * 0.8, y: Math.sin(angleA) * 0.8, name: 'Text A', label: textA },
        { x: Math.cos(angleB) * 0.8, y: Math.sin(angleB) * 0.8, name: 'Text B', label: textB },
    ]);
  };

  useEffect(() => {
    calculate();
  }, []);

  return (
    <div className="flex flex-col h-full bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
        <div className="flex items-center space-x-2 text-slate-700">
          <Ruler className="w-5 h-5 text-teal-600" />
          <span className="font-semibold text-sm">Step 3: 核心概念百科 - 向量空间 (Embedding Space)</span>
        </div>
        <div className="text-xs font-mono bg-teal-100 text-teal-700 px-2 py-1 rounded">
          def calculate_similarity(t1, t2):
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        
        {/* Input Area */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="space-y-2">
                <label className="text-xs font-bold text-slate-500 uppercase">Input Text A (Prediction)</label>
                <input 
                    type="text" 
                    value={textA}
                    onChange={(e) => setTextA(e.target.value)}
                    className="w-full p-3 border border-slate-200 rounded-lg focus:ring-2 focus:ring-teal-500 outline-none text-sm"
                />
            </div>
            <div className="space-y-2">
                <label className="text-xs font-bold text-slate-500 uppercase">Input Text B (Ground Truth)</label>
                <input 
                    type="text" 
                    value={textB}
                    onChange={(e) => setTextB(e.target.value)}
                    className="w-full p-3 border border-slate-200 rounded-lg focus:ring-2 focus:ring-teal-500 outline-none text-sm"
                />
            </div>
        </div>

        <button 
            onClick={calculate}
            className="w-full mb-8 py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-lg text-sm font-medium flex items-center justify-center transition-colors"
        >
            <RefreshCw className="w-4 h-4 mr-2" /> 重新计算相似度 (Recalculate)
        </button>

        {/* Visualization & Result */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            
            {/* Chart */}
            <div className="h-64 w-full bg-slate-50 rounded-xl border border-slate-100 relative">
                <div className="absolute top-2 left-2 text-xs text-slate-400 z-10">Vector Space Projection (2D)</div>
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="x" domain={[-1, 1]} hide />
                        <YAxis type="number" dataKey="y" domain={[-1, 1]} hide />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: chartData[0]?.x, y: chartData[0]?.y }]} stroke="#6366f1" strokeDasharray="3 3" />
                        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: chartData[1]?.x, y: chartData[1]?.y }]} stroke="#14b8a6" strokeDasharray="3 3" />
                        <Scatter name="Vectors" data={chartData} fill="#8884d8">
                            {chartData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={index === 0 ? '#6366f1' : '#14b8a6'} />
                            ))}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
                {/* Center Point */}
                <div className="absolute top-1/2 left-1/2 w-2 h-2 bg-slate-800 rounded-full -translate-x-1/2 -translate-y-1/2" title="Origin"></div>
            </div>

            {/* Metrics */}
            <div className="space-y-6">
                <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-slate-500 font-medium">Cosine Similarity</span>
                        <Calculator className="w-4 h-4 text-slate-400" />
                    </div>
                    <div className="flex items-end space-x-2">
                        <span className="text-4xl font-bold text-slate-800">{(similarity * 100).toFixed(1)}%</span>
                        <span className="text-sm text-slate-400 mb-1">Match Score</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-2 mt-3">
                        <div 
                            className="bg-gradient-to-r from-teal-400 to-indigo-500 h-2 rounded-full transition-all duration-700 ease-out"
                            style={{ width: `${similarity * 100}%` }}
                        ></div>
                    </div>
                </div>

                <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 text-sm text-blue-900">
                    <h4 className="font-bold mb-2">💡 什么是 Embedding?</h4>
                    <p className="leading-relaxed opacity-90">
                        计算机无法直接理解中文 "肝火上炎"。Embedding 将文字转换成一个长长的数字列表（向量），例如 <code>[0.12, -0.5, 0.88...]</code>。
                    </p>
                    <p className="mt-2 leading-relaxed opacity-90">
                        <b>余弦相似度</b> 通过计算两个向量夹角的余弦值来衡量它们在空间中的距离。夹角越小，含义越接近。
                    </p>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
};
