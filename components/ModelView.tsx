import React, { useState } from 'react';
import CodeHighlighter from './CodeHighlighter';
import { Layers, Zap, Code2, Calculator, MousePointer2, TrendingDown, Activity } from 'lucide-react';

export const ModelView: React.FC = () => {
  const [hoveredPart, setHoveredPart] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'code' | 'concepts'>('code');
  
  // Interactive Labs State
  const [reluInput, setReluInput] = useState<number>(-5);
  
  // Gradient Descent State
  const [learningRate, setLearningRate] = useState(0.1);
  const [w, setW] = useState(2.0); // Start at w=2
  const [gradHistory, setGradHistory] = useState<{w: number, loss: number}[]>([{w: 2.0, loss: 4.0}]);

  // Softmax State
  const [softmaxStep, setSoftmaxStep] = useState(0); // 0: input, 1: exp, 2: sum, 3: div

  const softmaxDemo = {
    logits: [2.0, 1.0],
    exps: [Math.exp(2.0), Math.exp(1.0)], // [7.389, 2.718]
    sum: Math.exp(2.0) + Math.exp(1.0), // 10.107
    probs: [Math.exp(2.0) / (Math.exp(2.0) + Math.exp(1.0)), Math.exp(1.0) / (Math.exp(2.0) + Math.exp(1.0))] // [0.73, 0.27]
  };

  // Gradient Descent Step
  const stepGradient = () => {
      const currentW = gradHistory[gradHistory.length - 1].w;
      const gradient = 2 * currentW; // Derivative of w^2 is 2w
      const newW = currentW - learningRate * gradient;
      const newLoss = newW * newW;
      
      setW(newW);
      setGradHistory([...gradHistory, {w: newW, loss: newLoss}]);
  };

  const resetGradient = () => {
      setW(2.0);
      setGradHistory([{w: 2.0, loss: 4.0}]);
  };

  return (
    <div className="h-full flex flex-col animate-fade-in-up">
      <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex justify-between items-start">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Layers className="text-purple-600" /> MLP 模型架构 (Model Architecture)
          </h2>
          <p className="text-slate-600 mt-2 text-sm">
             左侧是模型的“X光片”，右侧是其“基因图谱”（源代码）。点击右侧的“原理实验室”可以亲手验证数学原理。
          </p>
        </div>
      </div>

      <div className="flex-1 bg-white overflow-hidden flex flex-col lg:flex-row">
        
        {/* Left: Visualization */}
        <div className="flex-1 p-4 flex items-center justify-center bg-slate-50/30 relative">
          
          <div className="relative w-full max-w-2xl h-[500px] bg-white rounded-2xl border border-slate-200 shadow-sm p-4 flex items-center justify-between select-none">
            {/* Hover Guide */}
            <div className="absolute top-4 left-4 flex items-center gap-2 text-xs text-slate-400 bg-slate-50 px-2 py-1 rounded-full border border-slate-100">
               <MousePointer2 size={12} /> 悬停组件查看代码映射
            </div>

            {/* Input Layer */}
            <div 
               className="flex flex-col gap-4 items-center z-10 cursor-pointer group"
               onMouseEnter={() => setHoveredPart('input')}
               onMouseLeave={() => setHoveredPart(null)}
            >
               <div className={`text-xs font-bold mb-2 transition-colors ${hoveredPart === 'input' ? 'text-blue-600' : 'text-slate-500'}`}>Input (in_channels)</div>
               {[1, 2, 3].map(i => (
                 <div key={i} className={`w-12 h-12 rounded-full border-2 flex items-center justify-center bg-white transition-all ${hoveredPart === 'input' ? 'border-blue-500 scale-110 shadow-blue-200 shadow-lg' : 'border-slate-300 group-hover:border-slate-400'}`}>
                    <span className="text-[10px] font-mono text-slate-600">x_{i}</span>
                 </div>
               ))}
            </div>

            {/* Connection 1 */}
            <div className="flex-1 h-full relative mx-4">
                <svg className="w-full h-full absolute inset-0 pointer-events-none">
                    <line x1="0" y1="20%" x2="100%" y2="40%" stroke={hoveredPart === 'linear1' ? "#6366f1" : "#e2e8f0"} strokeWidth={hoveredPart === 'linear1' ? 3 : 1} className={hoveredPart === 'linear1' ? 'animate-flow' : ''} />
                    <line x1="0" y1="50%" x2="100%" y2="50%" stroke={hoveredPart === 'linear1' ? "#6366f1" : "#e2e8f0"} strokeWidth={hoveredPart === 'linear1' ? 3 : 1} className={hoveredPart === 'linear1' ? 'animate-flow' : ''} />
                    <line x1="0" y1="80%" x2="100%" y2="60%" stroke={hoveredPart === 'linear1' ? "#6366f1" : "#e2e8f0"} strokeWidth={hoveredPart === 'linear1' ? 3 : 1} className={hoveredPart === 'linear1' ? 'animate-flow' : ''} />
                </svg>
                <div 
                   className="absolute inset-0 flex items-center justify-center cursor-pointer group"
                   onMouseEnter={() => setHoveredPart('linear1')}
                   onMouseLeave={() => setHoveredPart(null)}
                >
                   <div className={`px-3 py-1.5 rounded-lg text-[10px] font-mono bg-white border shadow-sm transition-all flex flex-col items-center gap-1 ${hoveredPart === 'linear1' ? 'border-indigo-400 text-indigo-600 scale-110 bg-indigo-50 shadow-md' : 'border-slate-200 text-slate-400 hover:border-slate-300'}`}>
                      <span className="font-bold">Linear 1</span>
                      <span className="text-[8px] opacity-70">W1 * x + b1</span>
                   </div>
                </div>
            </div>

            {/* Hidden Layer (ReLU + Dropout) */}
            <div 
               className="flex flex-col gap-4 items-center z-10 cursor-pointer group"
               onMouseEnter={() => setHoveredPart('relu')}
               onMouseLeave={() => setHoveredPart(null)}
            >
               <div className={`text-xs font-bold mb-2 transition-colors ${hoveredPart === 'relu' ? 'text-purple-600' : 'text-slate-500'}`}>Hidden (ReLU)</div>
               {[1, 2, 3].map(i => (
                 <div key={i} className={`w-14 h-14 rounded-full border-2 flex items-center justify-center bg-white transition-all ${hoveredPart === 'relu' ? 'border-purple-500 scale-110 shadow-purple-200 shadow-lg' : 'border-purple-200 group-hover:border-purple-300'}`}>
                    <Zap size={20} className={hoveredPart === 'relu' ? 'fill-purple-600 text-purple-600 animate-pulse' : 'text-purple-300'} />
                 </div>
               ))}
            </div>

            {/* Connection 2 */}
            <div className="flex-1 h-full relative mx-4">
                <svg className="w-full h-full absolute inset-0 pointer-events-none">
                    <line x1="0" y1="40%" x2="100%" y2="20%" stroke={hoveredPart === 'linear2' ? "#6366f1" : "#e2e8f0"} strokeWidth={hoveredPart === 'linear2' ? 3 : 1} className={hoveredPart === 'linear2' ? 'animate-flow' : ''} />
                    <line x1="0" y1="50%" x2="100%" y2="50%" stroke={hoveredPart === 'linear2' ? "#6366f1" : "#e2e8f0"} strokeWidth={hoveredPart === 'linear2' ? 3 : 1} className={hoveredPart === 'linear2' ? 'animate-flow' : ''} />
                    <line x1="0" y1="60%" x2="100%" y2="80%" stroke={hoveredPart === 'linear2' ? "#6366f1" : "#e2e8f0"} strokeWidth={hoveredPart === 'linear2' ? 3 : 1} className={hoveredPart === 'linear2' ? 'animate-flow' : ''} />
                </svg>
                <div 
                   className="absolute inset-0 flex items-center justify-center cursor-pointer group"
                   onMouseEnter={() => setHoveredPart('linear2')}
                   onMouseLeave={() => setHoveredPart(null)}
                >
                   <div className={`px-3 py-1.5 rounded-lg text-[10px] font-mono bg-white border shadow-sm transition-all flex flex-col items-center gap-1 ${hoveredPart === 'linear2' ? 'border-indigo-400 text-indigo-600 scale-110 bg-indigo-50 shadow-md' : 'border-slate-200 text-slate-400 hover:border-slate-300'}`}>
                      <span className="font-bold">Linear 2</span>
                      <span className="text-[8px] opacity-70">W2 * h + b2</span>
                   </div>
                </div>
            </div>

            {/* Output Layer */}
             <div 
               className="flex flex-col gap-4 items-center z-10 cursor-pointer group"
               onMouseEnter={() => setHoveredPart('output')}
               onMouseLeave={() => setHoveredPart(null)}
            >
               <div className={`text-xs font-bold mb-2 transition-colors ${hoveredPart === 'output' ? 'text-emerald-600' : 'text-slate-500'}`}>Output (Softmax)</div>
               <div className={`w-12 h-12 rounded-full border-2 flex items-center justify-center bg-white transition-all ${hoveredPart === 'output' ? 'border-emerald-500 scale-110 shadow-emerald-200 shadow-lg' : 'border-slate-300 group-hover:border-slate-400'}`}>
                  <div className="flex flex-col items-center leading-none">
                     <span className="text-[10px] font-bold text-emerald-600">P(y)</span>
                     <span className="text-[8px] text-slate-400">0.73</span>
                  </div>
               </div>
               <div className={`w-12 h-12 rounded-full border-2 flex items-center justify-center bg-white transition-all ${hoveredPart === 'output' ? 'border-emerald-500 scale-110 shadow-emerald-200 shadow-lg' : 'border-slate-300 group-hover:border-slate-400'}`}>
                  <div className="flex flex-col items-center leading-none">
                     <span className="text-[10px] font-bold text-slate-400">P(y)</span>
                     <span className="text-[8px] text-slate-400">0.27</span>
                  </div>
               </div>
            </div>

          </div>
        </div>

        {/* Right: Code & Concepts */}
        <div className="w-full lg:w-[500px] bg-slate-900 border-l border-slate-700 shadow-xl flex flex-col h-[600px] lg:h-auto">
           
           {/* Tabs */}
           <div className="flex border-b border-slate-700 bg-slate-800 flex-shrink-0">
              <button 
                onClick={() => setActiveTab('code')}
                className={`flex-1 py-3 text-xs font-bold flex items-center justify-center gap-2 transition-colors ${activeTab === 'code' ? 'text-blue-400 bg-slate-900 border-t-2 border-blue-400' : 'text-slate-500 hover:text-slate-300 border-t-2 border-transparent'}`}
              >
                <Code2 size={14} /> 完整代码
              </button>
              <button 
                onClick={() => setActiveTab('concepts')}
                className={`flex-1 py-3 text-xs font-bold flex items-center justify-center gap-2 transition-colors ${activeTab === 'concepts' ? 'text-purple-400 bg-slate-900 border-t-2 border-purple-400' : 'text-slate-500 hover:text-slate-300 border-t-2 border-transparent'}`}
              >
                <Calculator size={14} /> 原理实验室
              </button>
           </div>

            {activeTab === 'code' ? (
              <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                {/* Use CodeHighlighter and map hoveredPart -> lines to highlight */}
                {(() => {
                  const codeLines = [
                    'import torch.nn as nn',
                    '',
                    'class MLP(torch.nn.Module):',
                    '    # 初始化：定义模型的各个层',
                    '    def __init__(self, in_channels, hidden_channels, out_channels, ...):',
                    '        super(MLP, self).__init__()',
                    '        # ModuleList 自动注册参数',
                    '        self.lins = torch.nn.ModuleList()',
                    '        # 第一层：输入 -> 隐藏',
                    '        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))',
                    '        # 中间隐藏层',
                    '        for _ in range(num_layers - 2):',
                    '            self.lins.append(...)',
                    '        # 最后一层：隐藏 -> 输出',
                    '        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))',
                    '        self.dropout = dropout',
                    '',
                    '    # 前向传播',
                    '    def forward(self, x):',
                    '        for i, lin in enumerate(self.lins[:-1]):',
                    '            x = lin(x)',
                    '            # 激活函数：增加非线性',
                    '            x = F.relu(x)',
                    '            x = F.dropout(x, p=self.dropout, ...)',
                    '        # 最后一层不需要 ReLU',
                    '        x = self.lins[-1](x)',
                    '        # 输出 Log 概率',
                    '        return F.log_softmax(x, dim=-1)'
                  ].join('\n');

                           const mapHoveredToLines = (part: string | null) => {
                              switch (part) {
                               case 'linear1': return [10,11]; // init first layer
                               case 'linear2': return [14,15,25]; // last layer and forward usage
                               case 'relu': return [22,23]; // activation lines
                               case 'output': return [26,27]; // return
                               default: return [];
                              }
                           };

                                             const modelTooltips: Record<number,string> = {
                                                 10: '第一层：输入到隐藏层的线性变换',
                                                 11: '向隐藏通道投影',
                                                 14: '最后一层：隐藏到输出',
                                                 22: '在前向传播中使用 ReLU 激活函数增加非线性',
                                                 23: '可选的 Dropout 用于防止过拟合',
                                                 26: '最终输出并计算 Log Softmax 概率',
                                             };

                                             // Build a highlightedMap so different parts use different colors
                                             const part = hoveredPart;
                                             const highlightedMap: Record<number,string> = {};
                                             if (part === 'linear1') {
                                                mapHoveredToLines('linear1').forEach(l => highlightedMap[l] = 'indigo');
                                             }
                                             if (part === 'linear2') {
                                                mapHoveredToLines('linear2').forEach(l => highlightedMap[l] = 'cyan');
                                             }
                                             if (part === 'relu') {
                                                mapHoveredToLines('relu').forEach(l => highlightedMap[l] = 'purple');
                                             }
                                             if (part === 'output') {
                                                mapHoveredToLines('output').forEach(l => highlightedMap[l] = 'emerald');
                                             }

                                             return <CodeHighlighter code={codeLines} highlightedMap={highlightedMap} lineTooltips={modelTooltips} dark={true} />;
                })()}
              </div>
           ) : (
              <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar bg-slate-900">
                 
                 {/* ReLU Lab */}
                 <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 shadow-lg">
                    <h4 className="text-purple-300 font-bold text-sm mb-4 flex items-center gap-2">
                       <Zap size={16}/> 
                       实验台 1: ReLU 激活函数
                    </h4>
                    <p className="text-slate-400 text-xs mb-4">
                       <strong>公式:</strong> <code>y = max(0, x)</code><br/>
                       试着拖动滑块，观察负数值如何被“砍断”归零。这是神经网络引入“非线性”的关键。
                    </p>
                    
                    {/* Interactive Slider */}
                    <div className="mb-6">
                       <div className="flex justify-between text-xs text-slate-500 mb-2 font-mono">
                          <span>Input: -10</span>
                          <span>Current: <span className="text-white font-bold">{reluInput}</span></span>
                          <span>Input: 10</span>
                       </div>
                       <input 
                         type="range" 
                         min="-10" 
                         max="10" 
                         step="0.5"
                         value={reluInput} 
                         onChange={(e) => setReluInput(parseFloat(e.target.value))}
                         className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                       />
                    </div>

                    {/* Visualization Graph */}
                    <div className="relative h-40 bg-slate-900 rounded border border-slate-700 overflow-hidden flex items-center justify-center">
                        {/* Grid Lines */}
                        <div className="absolute w-full h-[1px] bg-slate-600 bottom-5"></div> {/* X Axis */}
                        <div className="absolute h-full w-[1px] bg-slate-600 left-1/2"></div> {/* Y Axis */}
                        <div className="absolute left-1/2 bottom-1 -translate-x-1/2 text-[9px] text-slate-500">0</div>
                        
                        {/* Graph Line */}
                        <svg className="absolute top-0 left-0 w-full h-full pointer-events-none">
                            {/* ReLU Function: y = max(0, x) */}
                            {/* 
                                Viewport mapping:
                                X: 0% -> -10, 50% -> 0, 100% -> 10
                                Y: Bottom 20px (0), Top 0px (10)
                                Line Segment 1: x=0% to x=50%, y = bottom 20px (Val 0) - 水平线
                                Line Segment 2: x=50% to x=100%, y goes from bottom 20px to top 0px - 斜线
                            */}
                            {/* 水平部分：x < 0 时，y = 0 */}
                            <line x1="0%" y1="calc(100% - 20px)" x2="50%" y2="calc(100% - 20px)" stroke="#a855f7" strokeWidth="2" />
                            {/* 斜线部分：x >= 0 时，y = x */}
                            <line x1="50%" y1="calc(100% - 20px)" x2="100%" y2="0%" stroke="#a855f7" strokeWidth="2" />
                        </svg>

                        {/* The Point */}
                        <div 
                           className="absolute w-3 h-3 bg-white rounded-full border-2 border-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.8)] transition-all duration-75 z-10"
                           style={{
                              /* 
                                 Left: Map -10~10 to 0%~100%. Formula: (val + 10) / 20 * 100%
                                 Bottom: 
                                    Baseline is 20px. 
                                    Max Height above baseline is 100% - 20px.
                                    Val 0 -> Bottom 20px.
                                    Val 10 -> Bottom 100%.
                                    Fraction = val / 10.
                              */
                              left: `calc(${((reluInput + 10) / 20) * 100}% - 6px)`,
                              bottom: reluInput > 0 
                                ? `calc(20px + ${(reluInput/10)} * (100% - 20px) - 6px)` 
                                : `calc(20px - 6px)` 
                           }}
                        ></div>
                        
                        {/* Result Display */}
                        <div className="absolute top-2 left-2 text-xs font-mono bg-slate-800 px-2 py-1 rounded border border-slate-600 z-20">
                           Output: <span className={reluInput > 0 ? "text-emerald-400" : "text-rose-400"}>{Math.max(0, reluInput).toFixed(1)}</span>
                        </div>
                    </div>
                 </div>
                 
                 {/* Gradient Descent Lab */}
                 <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 shadow-lg">
                    <h4 className="text-blue-300 font-bold text-sm mb-4 flex items-center gap-2">
                       <TrendingDown size={16}/> 
                       实验台 2: 梯度下降 (优化器原理)
                    </h4>
                    <p className="text-slate-400 text-xs mb-4">
                       <strong>目标:</strong> 找到最低点 (Loss 最小)。<br/>
                       <strong>方法:</strong> 就像下山一样，顺着坡度(梯度)的方向迈步。步子太大容易滚过头，步子太小走得慢。
                    </p>

                    <div className="flex gap-4 mb-4">
                        <div className="flex-1">
                            <label className="text-[10px] text-slate-500 block mb-1">Learning Rate (步长)</label>
                            <input 
                                type="range" min="0.01" max="0.5" step="0.01" 
                                value={learningRate} onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                                className="w-full h-1 bg-slate-600 rounded cursor-pointer"
                            />
                            <div className="text-right text-[10px] text-blue-300 font-mono">{learningRate}</div>
                        </div>
                        <div className="flex items-end gap-2">
                             <button onClick={stepGradient} className="bg-blue-600 text-white px-3 py-1 rounded text-xs hover:bg-blue-700 active:scale-95 transition-all">
                                迈一步 (Step)
                             </button>
                             <button onClick={resetGradient} className="bg-slate-700 text-slate-300 px-3 py-1 rounded text-xs hover:bg-slate-600">
                                重置
                             </button>
                        </div>
                    </div>

                    <div className="relative h-40 bg-slate-900 rounded border border-slate-700 overflow-hidden">
                         {/* Graph y = w^2. w range -3 to 3. Loss range 0 to 9. */}
                         <svg className="absolute inset-0 w-full h-full" viewBox="-3 0 6 9" preserveAspectRatio="none">
                             {/* Note on SVG Coords:
                                viewBox="-3 0 6 9"
                                X: -3 (Left) to 3 (Right)
                                Y: 0 (Top) to 9 (Bottom)
                                We want Loss=0 at Bottom (SVG Y=9). Loss=9 at Top (SVG Y=0).
                                
                                Path Points:
                                (-3, 9) -> Loss (-3)^2 = 9. Map to Top. SVG Y = 0. Point: -3, 0.
                                (0, 0)  -> Loss 0. Map to Bottom. SVG Y = 9. Point: 0, 9.
                                (3, 9)  -> Loss 9. Map to Top. SVG Y = 0. Point: 3, 0.
                                
                                Quadratic Bezier Q Control Point:
                                To get vertex at (0,9) from (-3,0) and (3,0).
                                Midpoint t=0.5. Y = 0.25*0 + 0.5*Cy + 0.25*0 = 0.5*Cy.
                                We want Y=9. So 0.5*Cy = 9 => Cy = 18.
                                Control Point: (0, 18).
                             */}
                             
                             {/* Grid */}
                             <line x1="0" y1="0" x2="0" y2="9" stroke="#334155" strokeWidth="0.05" />
                             
                             {/* Curve: Valley Shape */}
                             <path d="M -3 0 Q 0 18 3 0" fill="none" stroke="#60a5fa" strokeWidth="0.1" vectorEffect="non-scaling-stroke" />
                             
                             {/* Ball */}
                             {/* 
                                cx = w. 
                                cy = Loss mapped to SVG Y.
                                Loss = w*w. 
                                SVG Y = 9 - w*w. (Since 0 is top, 9 is bottom, we invert).
                                Wait, based on the Path calculation above:
                                Loss 9 -> Y=0. Loss 0 -> Y=9.
                                SVG Y = 9 - Loss.
                             */}
                             <circle cx={w} cy={9 - w*w} r="0.2" fill="#f43f5e" className="transition-all duration-300" />
                         </svg>
                         <div className="absolute bottom-2 right-2 font-mono text-[10px] bg-slate-800/80 px-2 py-1 rounded text-slate-300">
                             Loss: {(w*w).toFixed(4)}
                         </div>
                    </div>
                 </div>

                 {/* Softmax Lab */}
                 <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 shadow-lg">
                    <h4 className="text-emerald-300 font-bold text-sm mb-4 flex items-center gap-2">
                       <Activity size={16}/> 
                       实验台 3: Softmax 计算器
                    </h4>
                    
                    <div className="flex gap-2 mb-4">
                       <button onClick={() => setSoftmaxStep(0)} className={`flex-1 text-[10px] py-1 rounded ${softmaxStep === 0 ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-400'}`}>1. Input</button>
                       <button onClick={() => setSoftmaxStep(1)} className={`flex-1 text-[10px] py-1 rounded ${softmaxStep === 1 ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-400'}`}>2. Exp</button>
                       <button onClick={() => setSoftmaxStep(2)} className={`flex-1 text-[10px] py-1 rounded ${softmaxStep === 2 ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-400'}`}>3. Sum</button>
                       <button onClick={() => setSoftmaxStep(3)} className={`flex-1 text-[10px] py-1 rounded ${softmaxStep === 3 ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-400'}`}>4. Div</button>
                    </div>

                    <div className="bg-slate-900 p-4 rounded border border-slate-700 font-mono text-xs space-y-3 min-h-[140px]">
                        {softmaxStep === 0 && (
                           <div className="animate-fade-in-up">
                              <div className="text-slate-500 mb-2"># 原始输出 Logits</div>
                              <div className="flex gap-4">
                                 <div className="bg-slate-800 p-2 rounded border border-slate-600 w-16 text-center">
                                    <div className="text-slate-400 text-[10px]">Index 0</div>
                                    <div className="text-white font-bold text-lg">2.0</div>
                                 </div>
                                 <div className="bg-slate-800 p-2 rounded border border-slate-600 w-16 text-center">
                                    <div className="text-slate-400 text-[10px]">Index 1</div>
                                    <div className="text-white font-bold text-lg">1.0</div>
                                 </div>
                              </div>
                           </div>
                        )}
                        {softmaxStep === 1 && (
                           <div className="animate-fade-in-up">
                              <div className="text-slate-500 mb-2"># Step 1: 取指数 (e^x)</div>
                              <div className="flex gap-4 items-center">
                                 <div className="text-center">
                                    <div className="text-slate-500 mb-1">e^2.0</div>
                                    <div className="bg-purple-900/50 p-2 rounded border border-purple-500/50 w-16 text-center text-purple-300 font-bold">
                                       {softmaxDemo.exps[0].toFixed(2)}
                                    </div>
                                 </div>
                                 <div className="text-center">
                                    <div className="text-slate-500 mb-1">e^1.0</div>
                                    <div className="bg-purple-900/50 p-2 rounded border border-purple-500/50 w-16 text-center text-purple-300 font-bold">
                                       {softmaxDemo.exps[1].toFixed(2)}
                                    </div>
                                 </div>
                              </div>
                           </div>
                        )}
                        {softmaxStep === 2 && (
                           <div className="animate-fade-in-up">
                              <div className="text-slate-500 mb-2"># Step 2: 求和 (分母)</div>
                              <div className="flex items-center gap-2 text-slate-300">
                                 <span className="text-purple-300">{softmaxDemo.exps[0].toFixed(2)}</span>
                                 <span>+</span>
                                 <span className="text-purple-300">{softmaxDemo.exps[1].toFixed(2)}</span>
                                 <span>=</span>
                                 <span className="bg-amber-900/50 px-2 py-1 rounded border border-amber-500/50 text-amber-300 font-bold">
                                    {softmaxDemo.sum.toFixed(2)}
                                 </span>
                              </div>
                           </div>
                        )}
                        {softmaxStep === 3 && (
                           <div className="animate-fade-in-up">
                              <div className="text-slate-500 mb-2"># Step 3: 归一化 (概率)</div>
                              <div className="space-y-2">
                                 <div className="flex items-center justify-between text-[10px]">
                                    <span className="text-slate-400">P(0) = {softmaxDemo.exps[0].toFixed(1)} / {softmaxDemo.sum.toFixed(1)}</span>
                                    <span className="text-emerald-400 font-bold text-sm">{softmaxDemo.probs[0].toFixed(2)} (73%)</span>
                                 </div>
                                 <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                     <div className="bg-emerald-500 h-full" style={{width: '73%'}}></div>
                                 </div>
                                 <div className="flex items-center justify-between text-[10px] mt-2">
                                    <span className="text-slate-400">P(1) = {softmaxDemo.exps[1].toFixed(1)} / {softmaxDemo.sum.toFixed(1)}</span>
                                    <span className="text-emerald-400 font-bold text-sm">{softmaxDemo.probs[1].toFixed(2)} (27%)</span>
                                 </div>
                                 <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                     <div className="bg-emerald-500 h-full" style={{width: '27%'}}></div>
                                 </div>
                              </div>
                           </div>
                        )}
                    </div>
                 </div>

              </div>
           )}

        </div>

      </div>
    </div>
  );
};