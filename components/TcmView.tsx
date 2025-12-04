
import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, FileJson, Sparkles, ArrowRight, BookOpen, ListVideo, Loader2, Terminal, Calculator, Clock, Play, Stethoscope, Quote, Scan, Brain, Check, RotateCcw } from 'lucide-react';

export const TcmView: React.FC<{ activeSubTab: string }> = ({ activeSubTab }) => {
  // --- State for Prompt Lab ---
  const [symptoms, setSymptoms] = useState("患者面色苍白，神疲乏力，少气懒言，舌淡苔白，脉虚无力。");
  const [promptStep, setPromptStep] = useState(0); 
  const [aiResponse, setAiResponse] = useState<string>("");

  // --- State for Batch Lab ---
  const [batchStatus, setBatchStatus] = useState<'idle' | 'running' | 'finished'>('idle');
  const [activeItem, setActiveItem] = useState(-1);
  const [processedItems, setProcessedItems] = useState<number[]>([]);
  
  // Track timers for cleanup to prevent white-screen crashes
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]); 
  const isMounted = useRef(true);

  const batchItems = [
    { id: 101, text: "发热，咽痛，舌红苔黄" },
    { id: 102, text: "头晕目眩，腰膝酸软" },
    { id: 103, text: "胃脘胀痛，嗳气吞酸" },
    { id: 104, text: "心悸失眠，纳差便溏" },
  ];

  // --- State for Eval Lab ---
  const [evalMode, setEvalMode] = useState<'calc' | 'space'>('calc'); 
  const [simScore, setSimScore] = useState<number | null>(null);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      clearTimers();
    };
  }, []);

  const clearTimers = () => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  };

  const handleBatchReset = () => {
      clearTimers();
      setBatchStatus('idle');
      setActiveItem(-1);
      setProcessedItems([]);
  };

  const startBatchProcess = () => {
      if (batchStatus === 'running') return;
      clearTimers();
      
      setBatchStatus('running');
      setProcessedItems([]);
      setActiveItem(-1);
      
      let currentIndex = 0;
      
      const processNext = () => {
          if (!isMounted.current) return;

          if (currentIndex >= batchItems.length) {
              setBatchStatus('finished');
              setActiveItem(-1);
              return;
          }

          setActiveItem(currentIndex);

          // Simulate API Call (Accelerated to 300ms)
          const t1 = setTimeout(() => {
              if (!isMounted.current) return;
              
              setProcessedItems(prev => [...prev, currentIndex]);
              currentIndex++;
              
              // Rate Limit Sleep (200ms)
              const t2 = setTimeout(() => {
                  if (!isMounted.current) return;
                  processNext();
              }, 200); 
              timersRef.current.push(t2);

          }, 300);
          timersRef.current.push(t1);
      };

      processNext();
  };

  // Prompt Simulation (or Real Call if Backend exists)
  const runPrompt = async () => {
     setPromptStep(1);
     setAiResponse("");
     
     // 1. Typing
     setTimeout(() => setPromptStep(2), 1000);

     // 2. Mocking Response (To use Real API, uncomment below)
     /*
     try {
       const res = await fetch('/api/gemini', {
          method: 'POST',
          body: JSON.stringify({ prompt: `System: ... User: ${symptoms}` })
       });
       const data = await res.json();
       setAiResponse(data.text);
     } catch(e) { console.error(e); }
     */

     // 3. Result
     setTimeout(() => {
        setAiResponse(`根据分析...\n\`\`\`json\n{\n  "证型": "气血两虚",\n  "治法": "补益气血"\n}\n\`\`\``);
        setPromptStep(3);
     }, 2500); 
  };

  // --- Render Functions ---

  const renderIntro = () => (
    <div className="h-full flex flex-col animate-fade-in-up overflow-hidden">
        <div className="p-4 border-b border-emerald-100 bg-emerald-50/30 flex-shrink-0">
          <h2 className="text-xl font-bold text-emerald-900 flex items-center gap-2">
            <BookOpen className="text-emerald-600" /> 0. 项目背景 (Context)
          </h2>
          <p className="text-emerald-700 mt-1 text-xs">
            慢性淋巴细胞白血病(CLL)的中医智能辨证。
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 bg-slate-50">
           <div className="max-w-4xl mx-auto space-y-6 pb-10">
               <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                   <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                       <Stethoscope className="text-rose-500" /> 医疗痛点
                   </h3>
                   <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                       <p>
                           <strong>慢性淋巴细胞白血病 (CLL)</strong> 症状复杂。传统诊疗面临数据混乱、标准不一等挑战。本项目旨在利用 <strong>大语言模型 (LLM)</strong> 实现从非结构化病历到结构化诊断的自动转换。
                       </p>
                       <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                           <div className="bg-slate-50 p-4 rounded border border-slate-100">
                               <div className="font-bold text-slate-700 mb-2 flex items-center gap-2"><Quote size={14}/> 输入：非结构化文本</div>
                               <div className="text-xs text-slate-500 italic bg-white p-2 rounded border border-slate-200">
                                   "患者自觉神疲乏力，午后低热，手足心热，盗汗，舌红少苔..."
                               </div>
                           </div>
                           <div className="bg-emerald-50 p-4 rounded border border-emerald-100">
                               <div className="font-bold text-emerald-700 mb-2 flex items-center gap-2"><Scan size={14}/> 输出：结构化 JSON</div>
                               <div className="text-xs text-emerald-600 font-mono bg-white p-2 rounded border border-emerald-200">
                                   {'{"证型": "气阴两虚", "治法": "益气养阴"}'}
                               </div>
                           </div>
                       </div>
                   </div>
               </div>
           </div>
        </div>
    </div>
  );

  const renderPrompt = () => (
    <div className="h-full flex flex-col lg:flex-row animate-fade-in-up overflow-hidden">
        {/* Top/Left: Visual Lab */}
        <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
             <div className="p-4 border-b border-emerald-100 bg-emerald-50/30">
                <h2 className="text-lg font-bold text-emerald-900 flex items-center gap-2">
                  <MessageSquare className="text-emerald-600" /> 1. 提示工程 (Prompt Engineering)
                </h2>
             </div>
             
             <div className="flex-1 overflow-y-auto p-4 space-y-6">
                {/* Input */}
                <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
                   <label className="text-xs font-bold text-slate-400 uppercase mb-2 block">Input Symptoms</label>
                   <textarea 
                      value={symptoms}
                      onChange={(e) => setSymptoms(e.target.value)}
                      className="w-full text-slate-700 bg-slate-50 border border-slate-200 rounded p-3 text-sm h-20 resize-none focus:ring-2 focus:ring-emerald-500 outline-none"
                   />
                   <div className="mt-2 flex justify-end">
                      <button 
                         onClick={runPrompt}
                         disabled={promptStep > 0 && promptStep < 3}
                         className="bg-emerald-600 text-white px-4 py-2 rounded text-xs font-bold flex items-center gap-2 hover:bg-emerald-700 disabled:opacity-50"
                      >
                         {promptStep > 0 && promptStep < 3 ? <Loader2 size={14} className="animate-spin"/> : <Sparkles size={14} />}
                         Generate
                      </button>
                   </div>
                </div>

                {/* System Prompt Visualization */}
                <div className={`transition-all duration-500 ${promptStep >= 1 ? 'opacity-100 translate-y-0' : 'opacity-50 translate-y-4 grayscale'}`}>
                   <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg text-xs font-mono">
                       <div className="text-blue-400 mb-2 font-bold">// Constructed Prompt sent to API</div>
                       <div className="text-slate-400">System:</div>
                       <div className="text-slate-300 ml-4 mb-2">你是一位经验丰富的中医专家，请根据患者症状描述判断：1. 证型 2. 治法...</div>
                       <div className="text-yellow-400 ml-4 mb-2">请用JSON格式输出：{`{"证型":"", "治法":""}`}</div>
                       <div className="text-slate-400">User:</div>
                       <div className="text-emerald-300 ml-4">{symptoms}</div>
                   </div>
                </div>

                {/* Response Extraction */}
                <div className={`transition-all duration-500 delay-100 ${promptStep >= 3 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                   <div className="flex flex-col md:flex-row gap-4">
                       <div className="flex-1 bg-white p-3 rounded-xl border border-slate-200 shadow-sm">
                           <div className="text-[10px] font-bold text-slate-400 uppercase mb-2">Raw Output</div>
                           <pre className="text-xs text-slate-600 font-mono whitespace-pre-wrap">{aiResponse}</pre>
                       </div>
                       <div className="flex items-center justify-center"><ArrowRight className="text-emerald-400 rotate-90 md:rotate-0"/></div>
                       <div className="flex-1 bg-emerald-50 p-3 rounded-xl border border-emerald-200 shadow-inner">
                           <div className="text-[10px] font-bold text-emerald-600 uppercase mb-2">Regex Parsed JSON</div>
                           <div className="text-sm font-bold text-emerald-800">
                               <div>证型: 气血两虚</div>
                               <div>治法: 补益气血</div>
                           </div>
                       </div>
                   </div>
                </div>
             </div>
        </div>

        {/* Bottom/Right: Code (Mobile: Bottom, Desktop: Right) */}
        <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
             <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-emerald-400 sticky top-0">
                <Terminal size={14} /> 
                <span className="text-xs font-bold">Source Code</span>
             </div>
             <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre custom-scrollbar">
{`# 1. 解析大模型响应
import re
import json

def parse_response(content):
    """
    解析大模型返回的内容，提取 JSON
    """
    # 正则表达式：匹配 \`\`\`json ... \`\`\` 之间的内容
    # re.DOTALL 让 . 也能匹配换行符
    code_block_pattern = re.compile(r"\`\`\`json\\s*(.*?)\\s*\`\`\`", re.DOTALL)
    
    match = code_block_pattern.search(content)

    if match:
        # 提取第一个捕获组的内容
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"JSON解析失败: {json_str}")
            return None
    else:
        # 如果没有代码块，尝试直接解析全文
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

# 2. 调用大模型 API
def call_large_model(symptoms):
    system_prompt = """
    你是一位经验丰富的中医专家，请根据患者症状描述判断：
    1. 证型：需符合慢性淋巴细胞白血病中医辨证标准
    2. 治法：需与证型对应且符合中医治疗原则
    请用JSON格式输出：{"证型":"", "治法":""}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"患者症状：{symptoms}"}
    ]

    try:
        # 使用 OpenAI SDK 或类似的 HTTP 客户端
        response = client.chat.completions.create(
            model="ernie-4.5-0.3b", # 或 gemini-pro
            messages=messages,
            max_tokens=512,
            temperature=0.3 # 低温度确保输出稳定
        )
        content = response.choices[0].message.content
        return parse_response(content)

    except Exception as e:
        print(f"API调用异常: {str(e)}")
        return {"证型": "未知", "治法": "待定"}
`}
             </div>
        </div>
    </div>
  );

  const renderBatch = () => (
    <div className="h-full flex flex-col lg:flex-row animate-fade-in-up overflow-hidden">
        <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
            <div className="p-4 border-b border-emerald-100 bg-emerald-50/30">
                <h2 className="text-lg font-bold text-emerald-900 flex items-center gap-2">
                    <ListVideo className="text-emerald-600" /> 2. 批量处理 (Batch)
                </h2>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 lg:p-8 flex flex-col items-center">
                {/* Controls */}
                <div className="w-full max-w-2xl bg-white p-4 rounded-xl border border-slate-200 shadow-sm mb-6 flex justify-between items-center">
                    <div>
                        <div className="text-sm font-bold text-slate-700">Dataset Queue</div>
                        <div className="text-xs text-slate-400">{batchItems.length} items to process</div>
                    </div>
                    <div className="flex gap-2">
                        <button onClick={handleBatchReset} className="px-3 py-1.5 text-xs font-bold text-slate-600 bg-slate-100 rounded hover:bg-slate-200 flex items-center gap-1">
                            <RotateCcw size={14}/> Reset
                        </button>
                        <button 
                            onClick={startBatchProcess} 
                            disabled={batchStatus === 'running' || batchStatus === 'finished'}
                            className="px-4 py-1.5 text-xs font-bold text-white bg-emerald-600 rounded hover:bg-emerald-700 flex items-center gap-2 disabled:opacity-50"
                        >
                            {batchStatus === 'running' ? <Loader2 size={14} className="animate-spin"/> : <Play size={14}/>}
                            Start Batch
                        </button>
                    </div>
                </div>

                {/* Conveyor Animation */}
                <div className="w-full max-w-3xl h-48 bg-slate-200 rounded-xl border border-slate-300 relative overflow-hidden flex items-center px-4 mb-6">
                    <div className="absolute inset-0 bg-[linear-gradient(90deg,transparent_49%,rgba(0,0,0,0.05)_50%,transparent_51%)] bg-[length:50px_100%] animate-[slide_1s_linear_infinite]"></div>
                    <style>{`@keyframes slide { from { background-position: 0 0; } to { background-position: 50px 0; } }`}</style>
                    
                    <div className="absolute left-1/2 -translate-x-1/2 top-0 bottom-0 w-24 border-x-2 border-emerald-400/30 bg-emerald-100/20 z-0 flex flex-col justify-center items-center">
                         <div className="text-[10px] font-bold text-emerald-600 uppercase tracking-widest bg-white/80 px-2 py-1 rounded">API Limit</div>
                         {activeItem > -1 && <Clock className="text-emerald-500 animate-pulse mt-2" size={20}/>}
                    </div>

                    <div className="flex gap-4 w-full justify-center relative z-10">
                        {batchItems.map((item, i) => {
                            let stateClass = "bg-white border-slate-300 text-slate-600";
                            let transClass = "";
                            
                            if (processedItems.includes(i)) {
                                stateClass = "bg-emerald-500 border-emerald-600 text-white opacity-0"; 
                                transClass = "translate-x-[200px]";
                            } else if (i === activeItem) {
                                stateClass = "bg-blue-600 border-blue-700 text-white shadow-lg scale-110 z-20";
                            }

                            return (
                                <div key={item.id} className={`w-24 h-32 rounded-lg border-2 p-2 flex flex-col transition-all duration-300 ${stateClass} ${transClass}`}>
                                    <div className="text-[10px] opacity-70 mb-1">#{item.id}</div>
                                    <div className="text-[10px] font-bold leading-tight line-clamp-3 flex-1">{item.text}</div>
                                    <div className="text-[9px] font-mono pt-2 border-t border-white/20 mt-1">
                                        {processedItems.includes(i) ? 'DONE' : (i === activeItem ? 'API...' : 'WAIT')}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Console */}
                <div className="w-full max-w-3xl bg-slate-900 rounded-xl p-4 h-32 overflow-y-auto custom-scrollbar font-mono text-xs border border-slate-700">
                    {batchStatus === 'idle' && <div className="text-slate-500">Ready to start...</div>}
                    {processedItems.map(idx => (
                        <div key={idx} className="text-emerald-400 mb-1 flex gap-2">
                             <Check size={12} className="mt-0.5"/> Saved result for ID {batchItems[idx].id}
                        </div>
                    ))}
                    {activeItem > -1 && (
                        <div className="text-amber-400 animate-pulse flex gap-2">
                            <Loader2 size={12} className="animate-spin mt-0.5"/> Processing ID {batchItems[activeItem].id}... (Sleep 0.2s)
                        </div>
                    )}
                    {batchStatus === 'finished' && <div className="text-blue-400 mt-2">[INFO] Batch job completed successfully.</div>}
                </div>
            </div>
        </div>

        {/* Code Panel */}
        <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
             <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-emerald-400 sticky top-0">
                <Terminal size={14} /> 
                <span className="text-xs font-bold">Source Code</span>
             </div>
             <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre custom-scrollbar">
{`import time
import csv

# 3. 批量处理循环
# 读取 CSV 数据
train_data = './datasets/train_data.csv'
symptoms_data = []

with open(train_data, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        symptoms_data.append(row[1]) # 提取症状列

predict_zx = []
predict_zf = []

# 遍历所有数据
for symptom in symptoms_data:
    # 调用大模型
    zx, zf = predict(symptom)
    
    predict_zx.append(zx)
    predict_zf.append(zf)
    
    # 关键：速率限制 (Rate Limiting)
    # 避免触发 API 的 QPS 限制 (429 Too Many Requests)
    time.sleep(0.2) 

print("Batch processing complete.")
`}
             </div>
        </div>
    </div>
  );

  const renderEval = () => (
    <div className="h-full flex flex-col lg:flex-row animate-fade-in-up overflow-hidden">
        <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
            <div className="p-4 border-b border-emerald-100 bg-emerald-50/30">
                <h2 className="text-lg font-bold text-emerald-900 flex items-center gap-2">
                    <Calculator className="text-emerald-600" /> 3. 效果评估 (Evaluation)
                </h2>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 lg:p-8">
                <div className="max-w-3xl mx-auto bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                    <div className="flex border-b border-slate-100">
                        <button onClick={() => setEvalMode('calc')} className={`flex-1 py-3 text-xs font-bold ${evalMode==='calc'?'bg-emerald-50 text-emerald-600 border-b-2 border-emerald-500':'text-slate-500'}`}>Cosine Similarity</button>
                        <button onClick={() => setEvalMode('space')} className={`flex-1 py-3 text-xs font-bold ${evalMode==='space'?'bg-purple-50 text-purple-600 border-b-2 border-purple-500':'text-slate-500'}`}>Embedding Space</button>
                    </div>

                    <div className="p-8 h-80 flex items-center justify-center">
                        {evalMode === 'calc' ? (
                            <div className="w-full max-w-md space-y-6">
                                <div className="flex justify-between items-center text-xs">
                                    <div className="bg-emerald-100 text-emerald-800 px-3 py-2 rounded">Pred: "气血两虚"</div>
                                    <div className="font-mono text-slate-400">VS</div>
                                    <div className="bg-blue-100 text-blue-800 px-3 py-2 rounded">Truth: "气虚血亏"</div>
                                </div>
                                <button 
                                    onClick={() => setSimScore(0.92)}
                                    className="w-full bg-slate-900 text-white py-2 rounded hover:bg-slate-800 transition-colors text-xs font-bold"
                                >
                                    Calculate Similarity
                                </button>
                                {simScore && (
                                    <div className="text-center animate-bounce">
                                        <div className="text-4xl font-black text-slate-800">{(simScore*100).toFixed(0)}%</div>
                                        <div className="text-xs text-emerald-500 font-bold mt-1">High Similarity</div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="relative w-64 h-64 bg-slate-50 border border-slate-200 rounded-lg">
                                <div className="absolute bottom-4 left-4 text-[10px] text-slate-400">0,0</div>
                                <svg className="w-full h-full absolute inset-0">
                                    <line x1="10%" y1="90%" x2="60%" y2="20%" stroke="#10b981" strokeWidth="2" markerEnd="url(#arrow)"/>
                                    <line x1="10%" y1="90%" x2="65%" y2="25%" stroke="#3b82f6" strokeWidth="2" markerEnd="url(#arrow)"/>
                                    <path d="M 50 150 Q 60 140 55 130" fill="none" stroke="#94a3b8" strokeDasharray="2"/>
                                    <defs>
                                        <marker id="arrow" markerWidth="4" markerHeight="4" refX="0" refY="2" orient="auto">
                                            <path d="M0,0 V4 L4,2 Z" fill="currentColor"/>
                                        </marker>
                                    </defs>
                                </svg>
                                <div className="absolute top-[20%] right-[30%] text-[9px] bg-emerald-100 text-emerald-800 px-1 rounded">Pred</div>
                                <div className="absolute top-[25%] right-[25%] text-[9px] bg-blue-100 text-blue-800 px-1 rounded">Truth</div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>

        {/* Code Panel */}
        <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
             <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-emerald-400 sticky top-0">
                <Terminal size={14} /> 
                <span className="text-xs font-bold">Source Code</span>
             </div>
             <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre custom-scrollbar">
{`from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 4. 获取 Embedding 并计算相似度

def get_embedding(text):
    """
    调用 Embeddings API 将文本转化为向量
    """
    url = "https://qianfan.baidubce.com/v2/embeddings"
    payload = json.dumps({
        "model": "bge-large-zh",
        "input": [text]
    })
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        data = response.json()
        # 返回 1024 维度的向量列表
        return data['data'][0]['embedding']
    except Exception as e:
        print("Embedding Error")
        return [0.0] * 1024

def calculate_similarity(text1, text2):
    """
    计算两个文本的余弦相似度
    """
    if not text1.strip() or not text2.strip():
        return 0.0
        
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    # Cosine Similarity Formula: (A . B) / (||A|| * ||B||)
    sim = cosine_similarity(
        np.array(emb1).reshape(1, -1),
        np.array(emb2).reshape(1, -1)
    )
    return float(sim[0][0])

def score(predict_zx, predict_zf):
    """
    最终打分逻辑
    """
    scores = []
    for i in range(len(predict_zx)):
        # 综合计算证型和治法的相似度
        sim_x = calculate_similarity(predict_zx[i], ZX[i])
        sim_f = calculate_similarity(predict_zf[i], ZF[i])
        scores.append((sim_x + sim_f) / 2)
        
    return np.mean(scores) * 100
`}
             </div>
        </div>
    </div>
  );

  switch (activeSubTab) {
    case 'TCM_INTRO': return renderIntro();
    case 'TCM_PROMPT': return renderPrompt();
    case 'TCM_BATCH': return renderBatch();
    case 'TCM_EVAL': return renderEval();
    default: return renderIntro();
  }
};
