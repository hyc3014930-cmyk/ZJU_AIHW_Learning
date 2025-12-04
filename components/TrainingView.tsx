import React, { useState, useEffect } from 'react';
import { Activity, Play, Pause, RotateCcw, FileCode, ChevronRight, BookOpen, Lightbulb, TrendingDown, TrendingUp } from 'lucide-react';

export const TrainingView: React.FC = () => {
  const TOTAL_EPOCHS = 5;
  const [epoch, setEpoch] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [step, setStep] = useState(0); 
  const [loss, setLoss] = useState(0.8);
  const [acc, setAcc] = useState(0.5);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setStep(prev => {
        if (prev < 4) return prev + 1;

        setEpoch(prevEpoch => {
          const nextEpoch = Math.min(prevEpoch + 1, TOTAL_EPOCHS);
          const progress = nextEpoch / TOTAL_EPOCHS;

          // 使用更陡峭的指数曲线，在 5 个 epoch 内完成收敛
          const easedLoss = 0.05 + 0.75 * Math.pow(1 - progress, 2.2);
          const easedAcc = 0.5 + 0.48 * Math.pow(progress, 1.3);
          setLoss(parseFloat(easedLoss.toFixed(4)));
          setAcc(parseFloat(easedAcc.toFixed(3)));

          if (nextEpoch === TOTAL_EPOCHS) {
            setIsRunning(false);
          }

          return nextEpoch;
        });

        return 1;
      });
    }, 1200);

    return () => clearInterval(interval);
  }, [isRunning]);

  const handleReset = () => {
    setIsRunning(false);
    setEpoch(0);
    setLoss(0.8);
    setAcc(0.5);
    setStep(0);
  };

  const stepDetails = {
    0: { 
      title: "准备阶段 Ready", 
      desc: "初始化权重与优化器，等同于 jirong.py → model.reset_parameters() + Adam。",
      concept: "所有参数尚未学习，Loss 很高，Accuracy 偶然命中。"
    },
    1: { 
      title: "1. 前向传播 (Forward)", 
      desc: "对应 jirong.py 中 train() 函数的 out = model(data.x[train_idx])。",
      concept: "将节点特征送入 MLP，得到 logits。"
    },
    2: { 
      title: "2. 计算损失 (Loss)", 
      desc: "loss = F.nll_loss(out, data.y[train_idx]) —— 与脚本一致。",
      concept: "衡量预测和标签差距，Loss 越小越好。"
    },
    3: { 
      title: "3. 反向传播 (Backward)", 
      desc: "loss.backward() 回传梯度，告诉每个参数该往哪走。",
      concept: "梯度累计在优化器中等待更新。"
    },
    4: { 
      title: "4. 参数更新 (Optimizer)", 
      desc: "optimizer.step() + optimizer.zero_grad()，与 jirong.py 完全一致。",
      concept: "Adam 用学习率 0.01 让权重快速下降，5 轮就收敛。"
    }
  };

  const currentDetail = stepDetails[step as keyof typeof stepDetails];

  return (
    <div className="h-full flex flex-col animate-fade-in-up overflow-hidden">
      <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 flex-shrink-0 z-10">
        <div>
           <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
             <Activity className="text-green-600" /> 3. 训练循环 (Training Loop)
           </h2>
        </div>
        <div className="flex gap-2">
           <button onClick={() => setIsRunning(!isRunning)} className="flex items-center gap-2 px-4 py-2 rounded bg-green-600 text-white text-xs font-bold">
              {isRunning ? <Pause size={14}/> : <Play size={14}/>} {isRunning ? 'Pause' : 'Start'}
           </button>
           <button onClick={handleReset} className="p-2 rounded border border-slate-200 hover:bg-slate-100">
              <RotateCcw size={14} className="text-slate-600"/>
           </button>
        </div>
      </div>

      <div className="flex-1 overflow-hidden flex flex-col lg:flex-row min-h-0">
         {/* Visualization */}
         <div className="flex-1 bg-slate-50/30 p-4 lg:p-8 flex flex-col gap-6 items-center overflow-y-auto">
            
            {/* Cycle Animation */}
            <div className="relative w-48 h-48 flex-shrink-0 select-none">
               <div className="absolute inset-0 rounded-full border-[8px] border-slate-200"></div>
               {[1,2,3,4].map(s => (
                   <div key={s} className={`absolute w-10 h-10 rounded-full flex items-center justify-center font-bold text-white transition-all duration-500 
                       ${step === s ? 'scale-110 opacity-100 bg-blue-600 shadow-lg ring-4 ring-blue-100' : 'opacity-40 bg-slate-400'}
                       ${s===1?'top-0 left-1/2 -translate-x-1/2 -translate-y-1/2':''}
                       ${s===2?'right-0 top-1/2 translate-x-1/2 -translate-y-1/2':''}
                       ${s===3?'bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2':''}
                       ${s===4?'left-0 top-1/2 -translate-x-1/2 -translate-y-1/2':''}
                   `}>
                       {s}
                   </div>
               ))}
               <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="text-4xl font-black text-slate-800">{epoch}</div>
                  <div className="text-[10px] uppercase font-bold text-slate-400">Epochs</div>
               </div>
            </div>

            {/* Info Card - Fixed Overflow */}
            <div className="w-full max-w-lg bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex-shrink-0">
               <div className="font-bold text-slate-800 mb-2 flex items-center gap-2">
                   <BookOpen size={16} className="text-blue-500"/> {currentDetail.title}
               </div>
               <p className="text-xs text-slate-600 mb-3">{currentDetail.desc}</p>
               <div className="bg-yellow-50 p-3 rounded border border-yellow-100 text-xs text-yellow-800 flex gap-2">
                   <Lightbulb size={16} className="flex-shrink-0"/>
                   {currentDetail.concept}
               </div>
            </div>

            {/* Metrics */}
            <div className="w-full max-w-lg flex gap-4">
                <div className="flex-1 bg-white p-3 rounded border border-slate-200">
                    <div className="text-[10px] text-slate-400 font-bold uppercase">Loss</div>
                    <div className="text-lg font-mono font-bold text-rose-600">{loss.toFixed(4)}</div>
                </div>
                <div className="flex-1 bg-white p-3 rounded border border-slate-200">
                    <div className="text-[10px] text-slate-400 font-bold uppercase">Accuracy</div>
                    <div className="text-lg font-mono font-bold text-emerald-600">{(acc*100).toFixed(1)}%</div>
                </div>
            </div>
            <div className="text-[11px] text-slate-500 text-center italic">
                * 为了演示效果，仅模拟 {TOTAL_EPOCHS} 次 epoch，Loss / Accuracy 都会在 5 步内快速收敛。
            </div>

         </div>

         {/* Code Panel */}
         <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
             <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-slate-400 sticky top-0">
                <FileCode size={14} />
                <span className="text-xs font-bold">jirong.py</span>
             </div>
             
             <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-loose relative custom-scrollbar whitespace-nowrap text-slate-400">
                <code className="block text-slate-500 mb-1"># 训练循环：与 jirong.py 完全一致</code>
                <div className={`transition-colors ${step===0?'text-white bg-emerald-900/40 px-2 -mx-2 rounded':''}`}>
                    model.reset_parameters()
                </div>
                <div className={`transition-colors ${step===0?'text-white bg-emerald-900/40 px-2 -mx-2 rounded':''}`}>
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-7)
                </div>
                <code className="block text-yellow-200 mt-3">def train(model, data, train_idx, optimizer):</code>
                <div className="pl-4 border-l border-slate-700 ml-1">
                    <div className={`transition-colors ${step===0?'text-white bg-emerald-900/40 px-2 -mx-2 rounded':''}`}>
                        model.train()
                    </div>
                    <div className={`transition-colors ${step===0?'text-white bg-emerald-900/40 px-2 -mx-2 rounded':''}`}>
                        optimizer.zero_grad()
                    </div>
                    <div className={`transition-colors ${step===1?'text-white bg-blue-900/50 px-2 -mx-2 rounded':''}`}>
                        out = model(data.x[train_idx])
                    </div>
                    <div className={`transition-colors ${step===2?'text-white bg-rose-900/50 px-2 -mx-2 rounded':''}`}>
                        loss = F.nll_loss(out, data.y[train_idx])
                    </div>
                    <div className={`transition-colors ${step===3?'text-white bg-amber-900/50 px-2 -mx-2 rounded':''}`}>
                        loss.backward()
                    </div>
                    <div className={`transition-colors ${step===4?'text-white bg-purple-900/50 px-2 -mx-2 rounded':''}`}>
                        optimizer.step()
                    </div>
                    <div>optimizer.zero_grad()</div>
                </div>
                <code className="block text-slate-500 mt-4"># 验证阶段 (test 函数节选)</code>
                <div className="pl-4 border-l border-slate-700 ml-1">
                    <code className="block text-slate-400">with torch.no_grad():</code>
                    <code className="block text-slate-400 pl-4">model.eval()</code>
                    <code className="block text-slate-400 pl-4">y_pred = out.exp()</code>
                    <code className="block text-slate-400 pl-4">eval_results[key] = evaluator.eval(...)</code>
                </div>
             </div>
         </div>

      </div>
    </div>
  );
};