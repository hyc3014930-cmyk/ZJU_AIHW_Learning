import React, { useState, useEffect } from 'react';
import { Activity, Play, Pause, RotateCcw, FileCode, ChevronRight, BookOpen, Lightbulb, TrendingDown, TrendingUp } from 'lucide-react';

export const TrainingView: React.FC = () => {
  const [epoch, setEpoch] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [step, setStep] = useState(0); // 0: Ready, 1: Forward, 2: Loss, 3: Backward, 4: Optimizer

  // Simulated metrics
  const [loss, setLoss] = useState(0.8);
  const [acc, setAcc] = useState(0.5);

  // Animation Loop
  useEffect(() => {
    let interval: any;
    if (isRunning && epoch < 200) {
      interval = setInterval(() => {
        // Cycle through steps 1->2->3->4
        setStep(prev => {
            if (prev < 4) return prev + 1;
            
            // End of cycle (Step 4 done), increment epoch and reset to Step 1
            setEpoch(e => e + 1);
            
            // Update metrics logic
            const progress = (epoch + 1) / 200;
            setLoss(0.8 * Math.exp(-4 * progress) + 0.1 + (Math.random() * 0.05));
            setAcc(0.5 + 0.45 * (1 - Math.exp(-3 * progress)));
            
            return 1; // Start new cycle immediately
        });
      }, 1500); // 1.5s per step
    }
    return () => clearInterval(interval);
  }, [isRunning, epoch]);

  const handleReset = () => {
    setIsRunning(false);
    setEpoch(0);
    setLoss(0.8);
    setAcc(0.5);
    setStep(0);
  };

  // Educational Content for each step
  const stepDetails = {
    0: {
      title: "准备就绪 (Ready)",
      desc: "模型已初始化，准备开始第一轮训练。点击“开始训练”观察循环过程。",
      concept: "训练就像刷题。Epoch 代表刷完了一整本习题集。我们需要刷很多遍 (200 Epochs) 才能掌握规律。"
    },
    1: {
      title: "1. 前向传播 (Forward)",
      desc: "代码: out = model(data.x) \n模型接收输入数据，经过层层计算，给出一个预测结果。",
      concept: "想象你在做一道选择题，凭直觉选了一个答案。这就是前向传播——不管对错，先给个结果。"
    },
    2: {
      title: "2. 计算误差 (Compute Loss)",
      desc: "代码: loss = F.nll_loss(out, data.y) \n将模型的预测结果与真实标签进行对比，计算差距。",
      concept: "Loss 函数（如 NLLLoss）就像老师批改作业。如果答案是 A，你选了 B，老师会告诉你'错得很离谱'（Loss 很大）；选了 A，就是'完全正确'（Loss 接近 0）。"
    },
    3: {
      title: "3. 反向传播 (Backward)",
      desc: "代码: loss.backward() \nPyTorch 自动计算梯度，寻找导致错误的'元凶'。",
      concept: "这是神经网络的核心魔法。Backward 就像从答案倒推解题步骤，找出哪一步算错了，算出每一层参数需要修改的方向（梯度）。"
    },
    4: {
      title: "4. 参数更新 (Optimizer Step)",
      desc: "代码: optimizer.step() \n优化器根据梯度信息，微调模型的所有参数。",
      concept: "知道了哪一步算错了还不够，必须改正。Optimizer (如 Adam) 根据反向传播回来的信息，稍微调整一下神经元的连接权重。下次再遇到同样的题，就不会犯同样的错了。"
    }
  };

  const currentDetail = stepDetails[step as keyof typeof stepDetails];

  return (
    <div className="h-full flex flex-col animate-fade-in-up">
      <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
        <div>
           <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
             <Activity className="text-green-600" /> 第三步：训练循环 (Training Loop)
           </h2>
           <p className="text-slate-600 mt-2 text-sm">
             观察右侧代码高亮，理解每一次“迭代”机器都做了什么。
           </p>
        </div>
        <div className="flex gap-2">
           <button 
             onClick={() => setIsRunning(!isRunning)} 
             className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold transition-colors ${isRunning ? 'bg-amber-100 text-amber-700 border border-amber-200' : 'bg-green-600 text-white shadow-lg shadow-green-200 hover:bg-green-700'}`}
           >
              {isRunning ? <Pause size={18} /> : <Play size={18} />}
              {isRunning ? '暂停讲解' : '开始训练'}
           </button>
           <button onClick={handleReset} className="flex items-center gap-2 bg-white text-slate-600 border border-slate-200 px-4 py-2 rounded-lg font-bold hover:bg-slate-50 transition-colors">
              <RotateCcw size={18} /> 重置
           </button>
        </div>
      </div>

      <div className="flex-1 overflow-hidden flex flex-col lg:flex-row">
         
         {/* Left: Dashboard Visualization */}
         <div className="flex-1 bg-slate-50/30 p-4 lg:p-8 flex flex-col gap-8 items-center justify-start overflow-y-auto">
            
            {/* 1. Status Cycle */}
            <div className="relative w-64 h-64 flex-shrink-0 mt-4 select-none">
               {/* Base Circle */}
               <div className="absolute inset-0 rounded-full border-[10px] border-slate-200"></div>
               
               {/* Step 1: Forward */}
               <div className={`absolute top-0 left-1/2 -translate-x-1/2 -translate-y-4 flex flex-col items-center transition-all duration-500 ${step === 1 ? 'scale-110 opacity-100 z-10' : 'opacity-40 scale-90 grayscale'}`}>
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white shadow-lg ${step === 1 ? 'bg-blue-600 ring-4 ring-blue-100' : 'bg-slate-400'}`}>1</div>
                  <div className="bg-white border px-2 py-1 rounded mt-2 text-xs font-bold text-slate-600 whitespace-nowrap shadow-sm">前向传播</div>
               </div>

               {/* Step 2: Loss */}
               <div className={`absolute right-0 top-1/2 translate-x-4 -translate-y-1/2 flex flex-col items-center transition-all duration-500 ${step === 2 ? 'scale-110 opacity-100 z-10' : 'opacity-40 scale-90 grayscale'}`}>
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white shadow-lg ${step === 2 ? 'bg-rose-600 ring-4 ring-rose-100' : 'bg-slate-400'}`}>2</div>
                  <div className="bg-white border px-2 py-1 rounded mt-2 text-xs font-bold text-slate-600 whitespace-nowrap shadow-sm">计算 Loss</div>
               </div>

               {/* Step 3: Backward */}
               <div className={`absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-4 flex flex-col-reverse items-center transition-all duration-500 ${step === 3 ? 'scale-110 opacity-100 z-10' : 'opacity-40 scale-90 grayscale'}`}>
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white shadow-lg ${step === 3 ? 'bg-amber-500 ring-4 ring-amber-100' : 'bg-slate-400'}`}>3</div>
                  <div className="bg-white border px-2 py-1 rounded mb-2 text-xs font-bold text-slate-600 whitespace-nowrap shadow-sm">反向传播</div>
               </div>

               {/* Step 4: Optimizer */}
               <div className={`absolute left-0 top-1/2 -translate-x-4 -translate-y-1/2 flex flex-col items-center transition-all duration-500 ${step === 4 ? 'scale-110 opacity-100 z-10' : 'opacity-40 scale-90 grayscale'}`}>
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white shadow-lg ${step === 4 ? 'bg-purple-600 ring-4 ring-purple-100' : 'bg-slate-400'}`}>4</div>
                  <div className="bg-white border px-2 py-1 rounded mt-2 text-xs font-bold text-slate-600 whitespace-nowrap shadow-sm">参数更新</div>
               </div>
               
               {/* Central Info */}
               <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="text-5xl font-black text-slate-800 tracking-tighter">{epoch}</div>
                  <div className="text-xs uppercase tracking-widest text-slate-400 font-bold mt-1">Epochs</div>
               </div>
            </div>

            {/* 2. Educational Concept Card (The "Why") */}
            <div className="w-full max-w-lg bg-white rounded-xl border border-slate-200 shadow-lg overflow-hidden transition-all duration-500">
               <div className={`px-4 py-3 border-b flex items-center gap-2 ${
                  step === 1 ? 'bg-blue-50 border-blue-100 text-blue-800' :
                  step === 2 ? 'bg-rose-50 border-rose-100 text-rose-800' :
                  step === 3 ? 'bg-amber-50 border-amber-100 text-amber-800' :
                  step === 4 ? 'bg-purple-50 border-purple-100 text-purple-800' :
                  'bg-slate-50 border-slate-200 text-slate-700'
               }`}>
                  <BookOpen size={18} />
                  <span className="font-bold text-sm">{currentDetail.title}</span>
               </div>
               <div className="p-5">
                  <p className="text-slate-600 text-sm mb-4 leading-relaxed font-medium whitespace-pre-line">
                     {currentDetail.desc}
                  </p>
                  <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-100 flex gap-3">
                     <Lightbulb className="text-yellow-600 flex-shrink-0 mt-0.5" size={18} />
                     <p className="text-xs text-yellow-800 leading-relaxed">
                        <strong className="block mb-1 text-yellow-900">原理画中画：</strong>
                        {currentDetail.concept}
                     </p>
                  </div>
               </div>
            </div>

            {/* 3. Metrics */}
            <div className="flex gap-4 w-full max-w-lg">
               <div className="flex-1 bg-white p-3 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
                  <div>
                     <div className="text-[10px] text-slate-400 font-bold uppercase mb-1">Training Loss</div>
                     <div className="flex items-center gap-2">
                        <div className="text-xl font-mono font-bold text-rose-600">{loss.toFixed(4)}</div>
                        <TrendingDown size={14} className="text-rose-500 animate-pulse"/>
                     </div>
                  </div>
                  <Activity size={20} className="text-rose-200" />
               </div>
               <div className="flex-1 bg-white p-3 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
                   <div>
                     <div className="text-[10px] text-slate-400 font-bold uppercase mb-1">AUC (Accuracy)</div>
                     <div className="flex items-center gap-2">
                        <div className="text-xl font-mono font-bold text-emerald-600">{(acc * 100).toFixed(1)}%</div>
                        <TrendingUp size={14} className="text-emerald-500 animate-pulse"/>
                     </div>
                  </div>
                  <Activity size={20} className="text-emerald-200" />
               </div>
            </div>

         </div>

         {/* Right: Code Panel with Active Highlighting */}
         <div className="w-full lg:w-[500px] bg-slate-900 border-l border-slate-700 flex flex-col shadow-2xl h-[600px] lg:h-auto">
             <div className="p-3 border-b border-slate-700 flex items-center gap-2 text-slate-400 bg-slate-900 z-10">
                <FileCode size={16} />
                <span className="text-sm font-bold">main.py (训练部分)</span>
             </div>
             
             <div className="flex-1 overflow-y-auto p-4 font-mono text-xs leading-loose relative custom-scrollbar">
                
                <code className="block text-slate-500 mb-2"># 1. 定义评估器和优化器</code>
                <code className="block text-slate-300">evaluator = Evaluator('auc')</code>
                <code className="block text-purple-300 mb-4">optimizer = torch.optim.Adam(model.parameters(), lr=0.01)</code>

                <code className="block text-slate-500 mb-2"># 2. 定义训练函数</code>
                <code className="block text-yellow-200">def train(model, data, train_idx, optimizer):</code>
                <div className="pl-4 border-l border-slate-700 ml-1">
                    <code className="block text-slate-300">model.train()</code>
                    <div className={`transition-all duration-300 ${step === 4 ? 'bg-purple-900/40 text-purple-200' : 'text-slate-500'}`}>
                        <code className="block">optimizer.zero_grad() <span className="text-slate-500 ml-2"># 清空上一步梯度</span></code>
                    </div>

                    <div className={`transition-all duration-500 rounded px-2 -mx-2 my-1 ${step === 1 ? 'bg-blue-900/60 border-l-2 border-blue-400 text-white py-1' : 'text-slate-300 opacity-60'}`}>
                        <div className="flex justify-between items-center">
                           <span>out = model(data.x[train_idx])</span>
                           {step === 1 && <ChevronRight size={14} className="text-blue-400"/>}
                        </div>
                    </div>

                    <div className={`transition-all duration-500 rounded px-2 -mx-2 my-1 ${step === 2 ? 'bg-rose-900/60 border-l-2 border-rose-400 text-white py-1' : 'text-slate-300 opacity-60'}`}>
                        <div className="flex justify-between items-center">
                           <span>loss = F.nll_loss(out, data.y[train_idx])</span>
                           {step === 2 && <ChevronRight size={14} className="text-rose-400"/>}
                        </div>
                    </div>

                    <div className={`transition-all duration-500 rounded px-2 -mx-2 my-1 ${step === 3 ? 'bg-amber-900/60 border-l-2 border-amber-400 text-white py-1' : 'text-slate-300 opacity-60'}`}>
                        <div className="flex justify-between items-center">
                           <span>loss.backward()</span>
                           {step === 3 && <ChevronRight size={14} className="text-amber-400"/>}
                        </div>
                    </div>

                    <div className={`transition-all duration-500 rounded px-2 -mx-2 my-1 ${step === 4 ? 'bg-purple-900/60 border-l-2 border-purple-400 text-white py-1' : 'text-slate-300 opacity-60'}`}>
                        <div className="flex justify-between items-center">
                           <span>optimizer.step()</span>
                           {step === 4 && <ChevronRight size={14} className="text-purple-400"/>}
                        </div>
                    </div>
                    <code className="block text-emerald-300 mt-1">return loss.item()</code>
                </div>
                
                <code className="block text-slate-500 mt-4 mb-2"># 3. 循环训练</code>
                <code className="block text-yellow-500">for epoch in range(1, epochs + 1):</code>
                <div className="pl-4 border-l border-slate-700 ml-1">
                    <code className="block text-slate-300">loss = train(model, data, ...)</code>
                    <code className="block text-slate-300">eval_results = test(model, data, ...)</code>
                    <code className="block text-slate-500">if epoch % 10 == 0: print(...)</code>
                </div>

             </div>
         </div>

      </div>
    </div>
  );
};