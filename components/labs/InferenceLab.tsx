import React, { useState } from 'react';
import { MessageSquare, ArrowDown, Bot, User, Play, Sparkles } from 'lucide-react';
import { MOCK_SYMPTOMS } from '../../constants';

export const InferenceLab: React.FC = () => {
  const [input, setInput] = useState(MOCK_SYMPTOMS[0]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ syndrome: string; treatment: string } | null>(null);

  const handleSimulate = () => {
    setLoading(true);
    setResult(null);
    
    // Simulate API delay
    setTimeout(() => {
      // Mock logic for demo purposes
      let mockRes = { syndrome: "未知", treatment: "建议复诊" };
      if (input.includes("脉弦数")) {
        mockRes = { syndrome: "肝火上炎证", treatment: "清肝泻火，凉血止血" };
      } else if (input.includes("气短懒言")) {
        mockRes = { syndrome: "脾肺气虚证", treatment: "补益脾肺" };
      } else if (input.includes("骨蒸潮热")) {
        mockRes = { syndrome: "肝肾阴虚证", treatment: "滋补肝肾" };
      }
      
      setResult(mockRes);
      setLoading(false);
    }, 1500);
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
        <div className="flex items-center space-x-2 text-slate-700">
          <Bot className="w-5 h-5 text-indigo-600" />
          <span className="font-semibold text-sm">Step 1: 大模型推理 (LLM Inference)</span>
        </div>
        <div className="text-xs font-mono bg-indigo-100 text-indigo-700 px-2 py-1 rounded">
          def call_large_model(symptoms):
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto space-y-6">
        {/* Prompt Construction Visual */}
        <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
          <div className="flex justify-between items-center mb-3">
             <div className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center">
                <MessageSquare className="w-3 h-3 mr-1" /> Prompt Construction
             </div>
             <button className="text-xs text-indigo-600 hover:underline">View Payload</button>
          </div>
          
          <div className="space-y-3">
            <div className="bg-white p-3 rounded-lg border border-purple-100 shadow-sm">
                <span className="text-xs font-bold text-purple-600 block mb-1">SYSTEM (角色设定)</span>
                <p className="text-sm text-slate-600 italic">"你是一位经验丰富的中医专家，请根据患者症状描述判断：1. 证型... 2. 治法... 请用JSON格式输出..."</p>
            </div>
            
            <div className="bg-white p-3 rounded-lg border border-blue-100 shadow-sm relative group">
                <span className="text-xs font-bold text-blue-600 block mb-1">USER (用户输入)</span>
                <div className="relative">
                    <textarea 
                        className="w-full text-sm text-slate-700 resize-none outline-none bg-transparent border-b border-dashed border-slate-300 focus:border-blue-400 p-1"
                        rows={3}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                    />
                    <div className="mt-2 flex gap-2 overflow-x-auto pb-1">
                        {MOCK_SYMPTOMS.map((s, i) => (
                            <button 
                                key={i}
                                onClick={() => setInput(s)}
                                className="text-[10px] whitespace-nowrap px-2 py-1 bg-slate-100 hover:bg-slate-200 rounded-full text-slate-600 transition-colors"
                            >
                                示例 {i + 1}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
          </div>
        </div>

        <div className="flex justify-center -my-2 relative z-10">
            <button 
                onClick={handleSimulate}
                disabled={loading}
                className={`flex items-center space-x-2 px-6 py-2 rounded-full text-white shadow-lg transition-all ${
                    loading ? 'bg-slate-400 cursor-not-allowed' : 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:scale-105 active:scale-95'
                }`}
            >
                {loading ? (
                    <>
                        <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                        <span>Reasoning...</span>
                    </>
                ) : (
                    <>
                        <Play className="w-4 h-4 fill-current" />
                        <span>运行推理 (Run Inference)</span>
                    </>
                )}
            </button>
        </div>

        {/* Response Visual */}
        <div className={`transition-all duration-500 ${result ? 'opacity-100 translate-y-0' : 'opacity-50 translate-y-4'}`}>
             <div className="bg-white p-1 rounded-2xl border-2 border-indigo-100 shadow-xl overflow-hidden">
                <div className="bg-gradient-to-r from-indigo-50 to-white p-4 border-b border-indigo-50 flex justify-between">
                    <span className="text-xs font-bold text-indigo-400 uppercase tracking-wider">API Response (Parsed JSON)</span>
                    {result && <span className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded-full flex items-center"><Sparkles className="w-3 h-3 mr-1"/> Success</span>}
                </div>
                
                {result ? (
                    <div className="p-6 grid grid-cols-1 gap-4 relative">
                        {/* Background Deco */}
                        <div className="absolute right-0 top-0 opacity-5 pointer-events-none">
                            <Bot className="w-40 h-40" />
                        </div>

                        <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg border-l-4 border-indigo-500">
                            <span className="text-sm font-semibold text-slate-500">SYNDROME (证型)</span>
                            <span className="text-lg font-bold text-indigo-700">{result.syndrome}</span>
                        </div>
                        <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg border-l-4 border-teal-500">
                            <span className="text-sm font-semibold text-slate-500">TREATMENT (治法)</span>
                            <span className="text-lg font-bold text-teal-700">{result.treatment}</span>
                        </div>
                    </div>
                ) : (
                    <div className="p-12 text-center text-slate-400 text-sm">
                        点击上方按钮运行模拟
                    </div>
                )}
             </div>
        </div>
        
        {/* Principle Explanation Card */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-900 mt-4">
            <h4 className="font-bold flex items-center mb-2">
                <span className="bg-amber-200 w-5 h-5 rounded-full flex items-center justify-center mr-2 text-xs">?</span>
                原理实验室：JSON Mode
            </h4>
            <p className="opacity-90 leading-relaxed">
                在代码中，我们通过 Prompt 要求模型 <code>"请用JSON格式输出"</code>，并编写了 <code>parse_response</code> 函数处理返回结果。这是因为 LLM 本质上是文本生成器，经常会包裹 Markdown 代码块（如 ```json ... ```），我们需要正则表达式提取纯净的 JSON 字符串才能被程序后续处理。
            </p>
        </div>

      </div>
    </div>
  );
};
