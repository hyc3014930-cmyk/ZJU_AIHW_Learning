
import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Settings, TrendingUp, Play, Pause, RotateCcw, Code } from 'lucide-react';
import { CodeLine } from '../types';

const LRVisualizer: React.FC = () => {
  // Parameters
  const [totalSteps, setTotalSteps] = useState(100);
  const [warmupSteps, setWarmupSteps] = useState(20);
  const [lrMax, setLrMax] = useState(0.1);
  const [decayType, setDecayType] = useState<'cosine' | 'square' | 'constant'>('cosine');
  
  // Execution State
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [chartData, setChartData] = useState<{ step: number; lr: number }[]>([]);
  const timerRef = useRef<number | null>(null);

  // Constants for calculation
  const lrInit = 0.0;
  const lrEnd = 0.001;

  // Code Definition
  const CODE_LINES: CodeLine[] = [
    { num: 1, code: `def build_lr(total_steps=${totalSteps}, warmup=${warmupSteps}):`, indent: 0 },
    { num: 2, code: "lr_all_steps = []", indent: 1 },
    { num: 3, code: "for i in range(total_steps):", indent: 1 },
    { num: 4, code: "if i < warmup_steps:", indent: 2 },
    { num: 5, code: "lr = lr_init + inc * (i + 1)  # Warmup", indent: 3 },
    { num: 6, code: "else:", indent: 2 },
    { num: 7, code: `if decay_type == '${decayType}':`, indent: 3 },
    { num: 8, code: "lr = calc_decay(lr_max, i) # Decay", indent: 4 },
    { num: 9, code: "lr_all_steps.append(lr)", indent: 2 },
    { num: 10, code: "return lr_all_steps", indent: 1 },
  ];

  // Helper to calculate full data for static background line
  const calculateFullData = () => {
    const data = [];
    const inc = warmupSteps > 0 ? (lrMax - lrInit) / warmupSteps : 0;
    const decaySteps = totalSteps - warmupSteps;

    for (let i = 0; i < totalSteps; i++) {
      let lr;
      if (i < warmupSteps) {
        lr = lrInit + inc * (i + 1);
      } else {
        const stepInDecay = i - warmupSteps;
        if (decayType === 'cosine') {
          const cosineDecay = 0.5 * (1 + Math.cos((Math.PI * stepInDecay) / decaySteps));
          lr = (lrMax - lrEnd) * cosineDecay + lrEnd;
        } else if (decayType === 'square') {
          const frac = 1 - stepInDecay / decaySteps;
          lr = (lrMax - lrEnd) * (frac * frac) + lrEnd;
        } else {
          lr = lrMax;
        }
      }
      data.push({ step: i, lr: Number(lr.toFixed(5)) });
    }
    return data;
  };

  const fullData = calculateFullData();

  // Reset simulation when parameters change
  useEffect(() => {
    resetSimulation();
  }, [totalSteps, warmupSteps, lrMax, decayType]);

  const resetSimulation = () => {
    setIsPlaying(false);
    if (timerRef.current) clearInterval(timerRef.current);
    setCurrentStep(0);
    setChartData([]);
  };

  const togglePlay = () => {
    if (isPlaying) {
      setIsPlaying(false);
      if (timerRef.current) clearInterval(timerRef.current);
    } else {
      setIsPlaying(true);
    }
  };

  // Animation Loop
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = window.setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false);
            if (timerRef.current) clearInterval(timerRef.current);
            return prev;
          }
          return prev + 1;
        });
      }, 50); // Speed of simulation
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, totalSteps]);

  // Sync Chart Data with Step
  useEffect(() => {
    setChartData(fullData.slice(0, currentStep + 1));
  }, [currentStep, totalSteps, warmupSteps, lrMax, decayType]);

  // Determine Active Lines
  const getActiveLines = () => {
    if (currentStep === 0 && chartData.length === 0) return [1, 2];
    if (currentStep >= totalSteps) return [10];
    
    const lines = [3]; // Always inside loop
    if (currentStep < warmupSteps) {
      lines.push(4, 5); // Warmup branch
    } else {
      lines.push(6, 7, 8); // Decay branch
    }
    lines.push(9); // Append
    return lines;
  };

  const activeLines = getActiveLines();

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-emerald-600" />
            LR Scheduler 实验室
          </h2>
          <p className="text-sm text-slate-500">动态模拟 <code className="bg-slate-100 px-1 rounded">build_lr</code> 函数的执行过程</p>
        </div>
        <div className="flex gap-2">
            <button
                onClick={togglePlay}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isPlaying 
                    ? 'bg-amber-100 text-amber-700 hover:bg-amber-200'
                    : 'bg-indigo-600 text-white hover:bg-indigo-700'
                }`}
            >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {isPlaying ? '暂停' : '开始执行'}
            </button>
            <button
                onClick={resetSimulation}
                className="p-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors border border-slate-200"
            >
                <RotateCcw className="w-5 h-5" />
            </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left: Code & Controls */}
        <div className="space-y-4">
            {/* Controls */}
            <div className="bg-slate-50 p-4 rounded-lg border border-slate-200 text-sm">
                <div className="flex items-center gap-2 font-semibold text-slate-700 mb-4 border-b pb-2">
                    <Settings className="w-4 h-4" /> 参数设置
                </div>
                <div className="space-y-4">
                    <div>
                        <div className="flex justify-between text-xs text-slate-500 mb-1">
                            <span>Total Steps: {totalSteps}</span>
                        </div>
                        <input 
                            type="range" min="50" max="200" step="10" 
                            value={totalSteps} onChange={(e) => setTotalSteps(Number(e.target.value))}
                            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                        />
                    </div>
                    <div>
                        <div className="flex justify-between text-xs text-slate-500 mb-1">
                            <span>Warmup Steps: {warmupSteps}</span>
                        </div>
                        <input 
                            type="range" min="0" max="50" step="5" 
                            value={warmupSteps} onChange={(e) => setWarmupSteps(Number(e.target.value))}
                            className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-orange-500"
                        />
                    </div>
                     <div>
                        <span className="text-xs text-slate-500 mb-1 block">Decay Strategy</span>
                        <div className="flex bg-white rounded border border-slate-200 p-0.5">
                             {['cosine', 'square', 'constant'].map(t => (
                                 <button
                                    key={t}
                                    onClick={() => setDecayType(t as any)}
                                    className={`flex-1 text-xs py-1 rounded ${decayType === t ? 'bg-indigo-50 text-indigo-700 font-medium' : 'text-slate-500'}`}
                                 >
                                     {t}
                                 </button>
                             ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Code View */}
            <div className="bg-slate-900 rounded-lg overflow-hidden border border-slate-800 shadow-inner">
                <div className="flex items-center gap-2 px-3 py-2 bg-slate-800 border-b border-slate-700">
                    <Code className="w-3 h-3 text-slate-400" />
                    <span className="text-xs font-mono text-slate-300">build_lr.py</span>
                </div>
                <div className="p-3 font-mono text-xs overflow-x-auto">
                    {CODE_LINES.map((line) => (
                        <div 
                            key={line.num} 
                            className={`flex py-0.5 px-1 rounded transition-colors duration-150 ${
                                activeLines.includes(line.num) ? 'bg-indigo-500/30 text-white' : 'text-slate-500'
                            }`}
                        >
                            <span className="w-6 text-slate-600 text-right mr-3 select-none">{line.num}</span>
                            <pre style={{ paddingLeft: `${line.indent * 12}px` }} className={`${activeLines.includes(line.num) ? 'text-indigo-200' : ''}`}>
                                {line.code}
                            </pre>
                        </div>
                    ))}
                </div>
            </div>
        </div>

        {/* Right: Chart Visualization */}
        <div className="lg:col-span-2 flex flex-col">
             <div className="flex-1 min-h-[350px] bg-slate-50 rounded-lg border border-slate-200 p-4 relative">
                <div className="absolute top-4 right-4 z-10 bg-white/90 backdrop-blur px-3 py-1.5 rounded-md border border-slate-200 shadow-sm text-xs font-mono">
                    <div>Step: <span className="font-bold text-indigo-600">{currentStep}</span> / {totalSteps}</div>
                    <div>LR: <span className="font-bold text-emerald-600">{chartData[currentStep]?.lr.toFixed(6) || '0.000000'}</span></div>
                </div>

                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={fullData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" vertical={false} />
                        <XAxis dataKey="step" hide />
                        <YAxis domain={[0, lrMax * 1.1]} hide />
                        <Tooltip content={() => null} />
                        {/* Background Guide Line */}
                        <Line 
                            type="monotone" 
                            dataKey="lr" 
                            stroke="#e2e8f0" 
                            strokeWidth={2} 
                            dot={false} 
                            isAnimationActive={false}
                        />
                        {/* Animated Line */}
                        <Line 
                            type="monotone" 
                            data={chartData}
                            dataKey="lr" 
                            stroke="#4f46e5" 
                            strokeWidth={3} 
                            dot={false}
                            isAnimationActive={false} // We manage animation manually via state
                        />
                        <ReferenceLine x={currentStep} stroke="#6366f1" strokeDasharray="3 3" />
                        {warmupSteps > 0 && <ReferenceLine x={warmupSteps} stroke="#f97316" strokeDasharray="3 3" label={{ value: 'Warmup', position: 'insideTopLeft', fill:'#f97316', fontSize:10 }} />}
                    </LineChart>
                </ResponsiveContainer>
             </div>
             
             {/* Log Output */}
             <div className="mt-4 p-3 bg-black text-green-400 font-mono text-xs rounded-lg h-24 overflow-y-auto shadow-inner border border-slate-800">
                <div className="opacity-50"># Training Log Output</div>
                {chartData.slice(-3).reverse().map((d, i) => (
                    <div key={d.step}>
                        Step {String(d.step).padStart(3, '0')}: LR = {d.lr.toFixed(6)} 
                        {d.step < warmupSteps ? <span className="text-orange-400 ml-2">[Warmup]</span> : <span className="text-blue-400 ml-2">[Decay]</span>}
                    </div>
                ))}
                {currentStep === 0 && <div>Ready to start...</div>}
             </div>
        </div>
      </div>
    </div>
  );
};

export default LRVisualizer;
