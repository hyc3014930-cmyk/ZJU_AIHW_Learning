
import React, { useState, useEffect } from 'react';
import { Lock, Unlock, Zap, Layers, ArrowRight, Code, Play, Trash2, Battery, Leaf, Recycle } from 'lucide-react';
import { CodeLine } from '../types';
import { GARBAGE_CATEGORIES } from '../constants';

// Flatten the categories for random selection
const GARBAGE_ITEMS = [
  { label: 'Battery', cn: '电池', category: 'Hazardous', cnCat: '有害垃圾', icon: Battery, color: 'text-red-600', ring: 'ring-red-200', border: 'border-red-500', bg: 'bg-red-50' },
  { label: 'Banana Peel', cn: '香蕉皮', category: 'Wet', cnCat: '湿垃圾', icon: Leaf, color: 'text-amber-700', ring: 'ring-amber-200', border: 'border-amber-500', bg: 'bg-amber-50' },
  { label: 'Cans', cn: '易拉罐', category: 'Recyclable', cnCat: '可回收物', icon: Recycle, color: 'text-blue-600', ring: 'ring-blue-200', border: 'border-blue-500', bg: 'bg-blue-50' },
  { label: 'Seashell', cn: '贝壳', category: 'Dry', cnCat: '干垃圾', icon: Trash2, color: 'text-slate-600', ring: 'ring-slate-200', border: 'border-slate-500', bg: 'bg-slate-50' },
  { label: 'Paint bucket', cn: '油漆桶', category: 'Hazardous', cnCat: '有害垃圾', icon: Battery, color: 'text-red-600', ring: 'ring-red-200', border: 'border-red-500', bg: 'bg-red-50' },
  { label: 'Newspaper', cn: '报纸', category: 'Recyclable', cnCat: '可回收物', icon: Recycle, color: 'text-blue-600', ring: 'ring-blue-200', border: 'border-blue-500', bg: 'bg-blue-50' },
];

const StructureVisualizer: React.FC = () => {
  const [frozen, setFrozen] = useState(true);
  const [isInferencing, setIsInferencing] = useState(false);
  const [inferenceStep, setInferenceStep] = useState(0); // 0: Idle, 1: Preprocess, 2: Backbone, 3: Head, 4: Result
  const [currentItem, setCurrentItem] = useState(GARBAGE_ITEMS[0]);

  const runInference = () => {
    if (isInferencing) return;
    // Pick a random item
    const randomItem = GARBAGE_ITEMS[Math.floor(Math.random() * GARBAGE_ITEMS.length)];
    setCurrentItem(randomItem);
    setIsInferencing(true);
    setInferenceStep(1);
  };

  useEffect(() => {
    if (isInferencing && inferenceStep > 0 && inferenceStep < 5) {
      const timer = setTimeout(() => {
        setInferenceStep(prev => prev + 1);
      }, 1200); // Slightly slower for better readability
      return () => clearTimeout(timer);
    } else if (inferenceStep === 5) {
        setIsInferencing(false);
        setInferenceStep(0);
    }
  }, [isInferencing, inferenceStep]);

  const INFERENCE_CODE: CodeLine[] = [
    { num: 1, code: "def construct(self, x):", indent: 0 },
    { num: 2, code: "# 1. MindSpore Preprocess", indent: 1 },
    { num: 3, code: "x = (x - mean) / std  # Normalization", indent: 1, highlight: inferenceStep === 1 },
    { num: 4, code: "# 2. Frozen Backbone", indent: 1 },
    { num: 5, code: "features = self.backbone(x) # 1280 dims", indent: 1, highlight: inferenceStep === 2 },
    { num: 6, code: "# 3. Fine-tuned Head", indent: 1 },
    { num: 7, code: "logits = self.head(features) # 26 classes", indent: 1, highlight: inferenceStep === 3 },
    { num: 8, code: "return argmax(logits)", indent: 1, highlight: inferenceStep === 4 },
  ];

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
          <Layers className="w-6 h-6 text-indigo-600" />
          MobileNetV2 推理演示
        </h2>
        <div className="flex items-center gap-3">
             <button
                onClick={runInference}
                disabled={isInferencing}
                className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                isInferencing 
                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
                    : 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm'
                }`}
            >
                <Play className="w-4 h-4" /> 开始推理 (Inference)
            </button>
            <div className="h-6 w-px bg-slate-300 mx-2"></div>
            <div className="flex bg-slate-100 p-1 rounded-lg">
                <button
                    onClick={() => setFrozen(true)}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                    frozen ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                    }`}
                >
                    冻结 Backbone
                </button>
                <button
                    onClick={() => setFrozen(false)}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                    !frozen ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                    }`}
                >
                    解冻 Backbone
                </button>
            </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Visualization Area */}
          <div className="lg:col-span-2 flex flex-col md:flex-row items-center justify-center gap-4 relative py-12 bg-slate-50/50 rounded-xl border border-slate-100 min-h-[300px]">
            {/* Input Image */}
            <div className={`flex flex-col items-center gap-2 transition-all duration-300 ${inferenceStep === 1 ? 'scale-110' : 'opacity-80'}`}>
                <div className={`w-24 h-24 bg-white rounded-xl border-2 border-dashed border-slate-300 flex flex-col items-center justify-center text-slate-500 overflow-hidden relative shadow-sm
                    ${inferenceStep === 1 ? 'ring-4 ring-emerald-100 border-emerald-500' : ''}
                `}>
                    <currentItem.icon className={`w-8 h-8 mb-1 ${currentItem.color}`} />
                    <span className="text-[10px] font-bold text-slate-400">{currentItem.cn}</span>
                </div>
                <div className="text-center">
                    <span className={`block text-xs font-mono font-bold ${inferenceStep === 1 ? 'text-emerald-600' : 'text-slate-400'}`}>Input Image</span>
                    {inferenceStep === 1 && <span className="text-[10px] text-emerald-500">224x224 RGB</span>}
                </div>
            </div>

            <ArrowRight className={`text-slate-300 transition-colors duration-300 ${inferenceStep === 1 ? 'text-emerald-500' : ''}`} />

            {/* Backbone */}
            <div className={`relative w-48 h-32 rounded-xl border-2 flex flex-col items-center justify-center transition-all duration-500 ${
            frozen 
                ? 'bg-sky-50 border-sky-200' 
                : 'bg-orange-50 border-orange-200'
            } ${inferenceStep === 2 ? 'ring-4 ring-emerald-100 scale-105 shadow-xl z-10' : ''}`}>
            <div className="absolute -top-3 left-4 bg-white px-2 text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1 border border-slate-100 rounded-full shadow-sm">
                Backbone
                {frozen ? <Lock className="w-3 h-3 text-sky-500" /> : <Unlock className="w-3 h-3 text-orange-500" />}
            </div>
            <span className={`text-base font-bold ${frozen ? 'text-sky-700' : 'text-orange-700'}`}>
                MobileNetV2
            </span>
            <span className="text-[10px] text-slate-500 mt-1 bg-white/50 px-2 rounded">1280 Feature Map</span>
            
            {/* Data flow animation inside */}
            {inferenceStep === 2 && (
                <div className="absolute inset-0 overflow-hidden rounded-xl">
                    <div className="absolute top-0 left-0 h-full w-2 bg-white/60 blur-md animate-[scan_1.2s_ease-in-out_infinite]" style={{ animationName: 'slide' }}></div>
                </div>
            )}
            </div>

            <ArrowRight className={`text-slate-300 transition-colors duration-300 ${inferenceStep === 2 ? 'text-emerald-500' : ''}`} />

            {/* Head */}
            <div className={`relative w-40 h-32 bg-indigo-50 border-2 border-indigo-200 rounded-xl flex flex-col items-center justify-center transition-all duration-300
                 ${inferenceStep === 3 ? 'ring-4 ring-emerald-100 scale-105 shadow-xl z-10 border-indigo-400' : ''}
            `}>
            <div className="absolute -top-3 left-4 bg-white px-2 text-[10px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1 border border-slate-100 rounded-full shadow-sm">
                Head
                <Zap className="w-3 h-3 text-yellow-500 fill-yellow-500" />
            </div>
            <span className="text-base font-bold text-indigo-700">Classifier</span>
            <span className="text-[10px] text-slate-500 mt-1 bg-white/50 px-2 rounded">FC Layer (26)</span>
             {inferenceStep === 3 && (
                 <div className="absolute bottom-3 w-full flex justify-center gap-1">
                     <span className="block w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce"></span>
                     <span className="block w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce delay-75"></span>
                     <span className="block w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce delay-150"></span>
                 </div>
            )}
            </div>
            
            <ArrowRight className={`text-slate-300 transition-colors duration-300 ${inferenceStep === 3 ? 'text-emerald-500' : ''}`} />

            {/* Output */}
            <div className={`flex flex-col items-center gap-2 transition-all duration-300 ${inferenceStep === 4 ? 'scale-110 opacity-100' : 'opacity-40'}`}>
                <div className={`w-24 h-24 rounded-xl border-2 flex flex-col items-center justify-center text-center p-1 shadow-sm
                    ${inferenceStep === 4 ? `${currentItem.bg} ${currentItem.border} ${currentItem.ring} ring-4 shadow-lg` : 'bg-slate-50 border-slate-200'}
                `}>
                    {inferenceStep === 4 ? (
                        <>
                            <span className={`text-xs font-bold ${currentItem.color} mb-1`}>{currentItem.cnCat}</span>
                            <span className="text-sm font-bold text-slate-800">{currentItem.cn}</span>
                            <span className="text-[9px] text-slate-400 mt-1">{currentItem.label}</span>
                        </>
                    ) : (
                        <span className="text-xs text-slate-400 font-bold">Waiting...</span>
                    )}
                </div>
                <span className={`text-xs font-mono ${inferenceStep === 4 ? 'text-emerald-600 font-bold' : 'text-slate-400'}`}>Prediction</span>
            </div>
          </div>

          {/* Code Panel */}
          <div className="bg-slate-900 rounded-xl overflow-hidden shadow-lg border border-slate-800 flex flex-col h-full min-h-[300px]">
             <div className="flex items-center gap-2 px-4 py-3 bg-slate-800 border-b border-slate-700">
                <Code className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-mono text-slate-300">inference.py</span>
                {isInferencing && <span className="ml-auto text-[10px] bg-emerald-900 text-emerald-300 px-2 py-0.5 rounded animate-pulse">Running...</span>}
            </div>
            <div className="flex-1 p-4 font-mono text-xs overflow-y-auto">
                {INFERENCE_CODE.map((line) => (
                     <div 
                        key={line.num} 
                        className={`flex py-1.5 px-1 rounded transition-colors duration-200 ${
                            line.highlight ? 'bg-emerald-900/40 border-l-2 border-emerald-500' : 'border-l-2 border-transparent'
                        }`}
                    >
                        <span className="w-6 text-slate-600 text-right mr-3 select-none">{line.num}</span>
                        <pre style={{ paddingLeft: `${line.indent * 12}px` }} className={`${line.highlight ? 'text-emerald-200' : 'text-slate-400'}`}>
                            {line.code}
                        </pre>
                    </div>
                ))}
            </div>
            <div className="p-3 bg-slate-800/50 border-t border-slate-700 text-[10px] text-slate-400 font-mono">
                {inferenceStep === 0 && "Ready to start MindSpore inference session..."}
                {inferenceStep === 1 && `> Processing input: ${currentItem.cn} (${currentItem.label})...`}
                {inferenceStep === 2 && "> MobileNetV2 Backbone extracting features..."}
                {inferenceStep === 3 && "> Classification Head calculating probabilities..."}
                {inferenceStep === 4 && `> Result: Class '${currentItem.cn}' (Confidence: ${(90 + Math.random() * 9).toFixed(1)}%)`}
            </div>
          </div>
      </div>
    </div>
  );
};

export default StructureVisualizer;
