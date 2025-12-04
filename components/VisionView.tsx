import React, { useState, useEffect } from 'react';
import { Image as ImageIcon, Layers, Cpu, ArrowRight, Settings, Box, BarChart3, ScanLine, Terminal, BookOpen, Trash2, Zap, Search, Loader2, History, Scaling, Grid, Minimize, Play, Pause, RotateCcw, Activity, FileCode, Database, TrendingDown, Calculator } from 'lucide-react';

export const VisionView: React.FC<{ activeSubTab: string }> = ({ activeSubTab }) => {
  const [pipelineStep, setPipelineStep] = useState(0); 
  const [lrType, setLrType] = useState<'cosine' | 'square' | 'constant'>('cosine');
  const [inferState, setInferState] = useState(0);
  const [archLabMode, setArchLabMode] = useState<'struct' | 'conv' | 'pool'>('struct');
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0.8);
  const [featureExtractStep, setFeatureExtractStep] = useState(0);
  
  // Animation states
  const [convKernelPos, setConvKernelPos] = useState({ row: 0, col: 0 });
  const [isConvAnimating, setIsConvAnimating] = useState(false);
  const [poolingStep, setPoolingStep] = useState(0); 

  // --- Render Functions ---

  const renderIntro = () => (
    <div className="h-full flex flex-col animate-fade-in-up overflow-hidden">
        <div className="p-4 border-b border-cyan-100 bg-cyan-50/30 flex-shrink-0">
        <h2 className="text-xl font-bold text-cyan-900 flex items-center gap-2">
            <BookOpen className="text-cyan-600" /> 0. 项目背景 (Background)
        </h2>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 bg-slate-50">
            <div className="max-w-4xl mx-auto space-y-6 pb-10">
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                    <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                        <History size={20} className="text-blue-500" /> 从 LeNet5 到 MobileNetV2
                    </h3>
                    <div className="flex flex-col md:flex-row gap-4">
                        <div className="flex-1 bg-slate-50 p-4 rounded-xl border border-slate-200 text-center">
                            <div className="text-2xl mb-2">🔢 LeNet5</div>
                            <p className="text-xs text-slate-500">Deep Learning "Hello World". MNIST (28x28). Simple CNN.</p>
                        </div>
                        <div className="flex-1 bg-cyan-50 p-4 rounded-xl border border-cyan-200 text-center ring-2 ring-cyan-100">
                            <div className="text-2xl mb-2">🚀 MobileNetV2</div>
                            <p className="text-xs text-slate-500">Industrial Standard. Efficient, Lightweight. ImageNet Pre-trained.</p>
                        </div>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                    <h3 className="font-bold text-slate-700 mb-4 flex items-center gap-2">
                        <Trash2 size={20} className="text-emerald-500" /> 垃圾分类任务
                    </h3>
                    <div className="grid grid-cols-4 gap-2 text-xs text-slate-600">
                        {['Plastic Bottle', 'Hats', 'Newspaper', 'Cans', 'Glassware', 'Cardboard', 'Banana Peel', 'Battery'].map(n => (
                            <div key={n} className="bg-slate-50 p-2 rounded border border-slate-100 text-center">{n}</div>
                        ))}
                        <div className="col-span-4 text-center text-slate-400 mt-2">... Total 26 Classes</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
  );

  const renderPipeline = () => (
    <div className="h-full flex flex-col lg:flex-row animate-fade-in-up overflow-hidden">
      <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
          <div className="p-4 border-b border-cyan-100 bg-cyan-50/30">
            <h2 className="text-lg font-bold text-cyan-900 flex items-center gap-2">
              <ImageIcon className="text-cyan-600" /> 1. 数据流水线 (Pipeline)
            </h2>
          </div>

          <div className="flex-1 overflow-y-auto p-4 flex flex-col items-center gap-6">
               <div className="flex gap-2 mb-4">
                   {[0,1,2,3].map(i => (
                       <button 
                         key={i} 
                         onClick={() => setPipelineStep(i)}
                         className={`w-10 h-10 rounded-full font-bold text-sm border-2 ${pipelineStep === i ? 'border-cyan-500 bg-white text-cyan-600' : 'border-slate-200 bg-slate-100 text-slate-400'}`}
                       >
                           {i+1}
                       </button>
                   ))}
               </div>

               <div className="w-64 h-64 bg-white border border-slate-200 rounded-xl shadow-lg flex items-center justify-center relative perspective-1000">
                    {pipelineStep === 0 && <div className="text-center"><ImageIcon size={48} className="mx-auto text-slate-400 mb-2"/><div className="font-bold text-slate-700">Raw Image (RGB)</div></div>}
                    {pipelineStep === 1 && <div className="text-center"><Scaling size={48} className="mx-auto text-blue-500 mb-2"/><div className="font-bold text-slate-700">Resize (224x224)</div></div>}
                    {pipelineStep === 2 && <div className="text-center"><div className="grid grid-cols-3 gap-1 w-16 h-16 mx-auto mb-2 opacity-50"><div className="bg-purple-400"></div><div className="bg-purple-400"></div><div className="bg-purple-400"></div></div><div className="font-bold text-slate-700">Normalize</div></div>}
                    {pipelineStep === 3 && (
                        <div className="transform-style-3d animate-[spinY_4s_infinite]">
                            <style>{`@keyframes spinY { 0% { transform: rotateY(0deg); } 100% { transform: rotateY(360deg); } }`}</style>
                            <div className="w-20 h-24 border-2 border-red-400 bg-red-100/50 absolute -translate-x-4 translate-z-10 flex items-center justify-center text-red-500 font-bold">R</div>
                            <div className="w-20 h-24 border-2 border-green-400 bg-green-100/50 absolute flex items-center justify-center text-green-500 font-bold">G</div>
                            <div className="w-20 h-24 border-2 border-blue-400 bg-blue-100/50 absolute translate-x-4 -translate-z-10 flex items-center justify-center text-blue-500 font-bold">B</div>
                        </div>
                    )}
               </div>
               <div className="text-center text-sm font-bold text-slate-600">
                   {pipelineStep===0 && "Original Input"}
                   {pipelineStep===1 && "Bilinear Interpolation"}
                   {pipelineStep===2 && "(Pixel - Mean) / Std"}
                   {pipelineStep===3 && "HWC -> CHW Transpose"}
               </div>
          </div>
      </div>

      <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
            <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-cyan-400 sticky top-0">
               <Terminal size={14} /> <span className="text-xs font-bold">Source Code</span>
            </div>
            <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre custom-scrollbar">
{`# 图像预处理流水线
from PIL import Image
import numpy as np
import mindspore as ms
from mindspore import Tensor

def image_process(image):
    """
    Args:
        image: shape (H, W, C), 0-255
    """
    # 1. 确保 RGB 格式
    img = Image.fromarray(image).convert('RGB')

    # 2. Resize 到 224x224 (MobileNet 标准输入)
    img = img.resize((224, 224), Image.BILINEAR)

    # 3. 归一化 (Normalize)
    # 先转为 float32 并除以 255 -> [0, 1]
    img = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet 标准均值和方差
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # 4. 维度变换 (HWC -> CHW)
    # MindSpore/PyTorch 使用 Channel-First 格式
    img = img.transpose(2, 0, 1)

    # 5. 增加 Batch 维度 -> (1, 3, 224, 224)
    return Tensor(img[None, ...], ms.float32)
`}
            </div>
      </div>
    </div>
  );

  const renderArch = () => (
    <div className="h-full flex flex-col lg:flex-row animate-fade-in-up overflow-hidden">
       <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
            <div className="p-4 border-b border-cyan-100 bg-cyan-50/30">
                <h2 className="text-lg font-bold text-cyan-900 flex items-center gap-2">
                    <Layers className="text-cyan-600" /> 2. 模型架构 (Architecture)
                </h2>
            </div>
            <div className="flex-1 overflow-y-auto p-4 flex flex-col items-center">
                 <div className="flex gap-2 mb-6">
                     <button onClick={() => setArchLabMode('struct')} className={`px-3 py-1 text-xs font-bold rounded ${archLabMode==='struct'?'bg-cyan-100 text-cyan-700':'bg-slate-100 text-slate-500'}`}>Structure</button>
                     <button onClick={() => setArchLabMode('conv')} className={`px-3 py-1 text-xs font-bold rounded ${archLabMode==='conv'?'bg-purple-100 text-purple-700':'bg-slate-100 text-slate-500'}`}>Conv Lab</button>
                     <button onClick={() => setArchLabMode('pool')} className={`px-3 py-1 text-xs font-bold rounded ${archLabMode==='pool'?'bg-emerald-100 text-emerald-700':'bg-slate-100 text-slate-500'}`}>Pooling</button>
                 </div>

                 {archLabMode === 'struct' ? (
                     <div className="space-y-6 w-full max-w-2xl">
                         {/* Feature Extraction Visualization */}
                         <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                             <h3 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                                 <Database size={16} className="text-cyan-500"/> 特征提取流程 (Feature Extraction)
                             </h3>
                             <div className="flex items-center gap-4 mb-4 relative">
                                 {/* Data Flow Animation */}
                                 <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                     <div className="w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-[flow_2s_ease-in-out_infinite] opacity-60"></div>
                                 </div>
                                 <style>{`
                                     @keyframes flow {
                                         0% { transform: translateX(-100%); opacity: 0; }
                                         50% { opacity: 1; }
                                         100% { transform: translateX(100%); opacity: 0; }
                                     }
                                 `}</style>
                                 
                                 <div className="flex-1 bg-slate-50 p-3 rounded border border-slate-200 relative z-10 transition-all duration-500 hover:scale-105 hover:shadow-md">
                                     <div className="text-[10px] text-slate-400 mb-1">Input Image</div>
                                     <div className="text-xs font-mono text-slate-700">(1, 3, 224, 224)</div>
                                     <div className="absolute -top-1 -right-1 w-3 h-3 bg-cyan-400 rounded-full animate-ping opacity-75"></div>
                                 </div>
                                 <ArrowRight className="text-slate-300 z-10 animate-pulse"/>
                                 <div className="flex-1 bg-cyan-50 p-3 rounded border border-cyan-200 relative z-10 transition-all duration-500 hover:scale-105 hover:shadow-md">
                                     <div className="text-[10px] text-cyan-600 mb-1">Backbone (Frozen)</div>
                                     <div className="text-xs font-mono text-cyan-700">MobileNetV2</div>
                                     <div className="absolute inset-0 bg-cyan-200/20 rounded animate-pulse"></div>
                                 </div>
                                 <ArrowRight className="text-slate-300 z-10 animate-pulse"/>
                                 <div className="flex-1 bg-purple-50 p-3 rounded border border-purple-200 relative z-10 transition-all duration-500 hover:scale-105 hover:shadow-md">
                                     <div className="text-[10px] text-purple-600 mb-1">Feature Map</div>
                                     <div className="text-xs font-mono text-purple-700">(1, 1280, 7, 7)</div>
                                 </div>
                             </div>
                             
                             {/* Feature Map Visualization with Animation */}
                             <div className="relative">
                                 <div className="grid grid-cols-7 gap-1 bg-slate-900 p-2 rounded relative overflow-hidden">
                                     {/* Animated overlay */}
                                     <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-400/30 to-transparent animate-[sweep_3s_ease-in-out_infinite] pointer-events-none"></div>
                                     <style>{`
                                         @keyframes sweep {
                                             0% { transform: translateX(-100%) skewX(-20deg); }
                                             100% { transform: translateX(200%) skewX(-20deg); }
                                         }
                                     `}</style>
                                     
                                     {Array.from({length: 49}).map((_, i) => {
                                         const row = Math.floor(i / 7);
                                         const col = i % 7;
                                         const delay = (row + col) * 0.05;
                                         return (
                                             <div 
                                                 key={i} 
                                                 className="aspect-square bg-gradient-to-br from-cyan-400 to-purple-500 opacity-60 rounded-sm transition-all duration-300 hover:opacity-100 hover:scale-110"
                                                 style={{ 
                                                     opacity: 0.3 + (i % 7) * 0.1,
                                                     animation: `fadeIn 0.5s ease-out ${delay}s both`
                                                 }}
                                             />
                                         );
                                     })}
                                 </div>
                                 <style>{`
                                     @keyframes fadeIn {
                                         from { opacity: 0; transform: scale(0.8); }
                                         to { opacity: 1; transform: scale(1); }
                                     }
                                 `}</style>
                                 <div className="text-[10px] text-slate-500 mt-2 text-center">7x7 Feature Map (1280 channels)</div>
                             </div>
                         </div>

                         <div className="space-y-4">
                             <div className="bg-slate-200 p-4 rounded-lg border border-slate-300 opacity-70 relative transition-all duration-500 hover:opacity-90 hover:scale-105">
                                 <div className="absolute top-2 right-2 text-[10px] bg-slate-400 text-white px-1 rounded">Frozen</div>
                                 <div className="font-bold text-slate-700">Backbone</div>
                                 <div className="text-xs text-slate-500">MobileNetV2 (ImageNet Pre-trained)</div>
                                 <div className="text-[10px] font-mono mt-1">Out: 1280 channels @ 7x7</div>
                                 {/* Lock icon animation */}
                                 <div className="absolute bottom-2 right-2 w-4 h-4 border-2 border-slate-500 rounded animate-pulse"></div>
                             </div>
                             <div className="flex justify-center relative">
                                 <ArrowRight className="text-slate-300 rotate-90 animate-bounce"/>
                                 <div className="absolute inset-0 flex items-center justify-center">
                                     <div className="w-8 h-1 bg-gradient-to-r from-transparent via-slate-300 to-transparent animate-[flowVertical_1.5s_ease-in-out_infinite]"></div>
                                 </div>
                                 <style>{`
                                     @keyframes flowVertical {
                                         0% { transform: translateY(-10px); opacity: 0; }
                                         50% { opacity: 1; }
                                         100% { transform: translateY(10px); opacity: 0; }
                                     }
                                 `}</style>
                             </div>
                             <div className="bg-white p-4 rounded-lg border-2 border-rose-400 relative shadow-lg transition-all duration-500 hover:scale-105 hover:shadow-xl">
                                 <div className="absolute top-2 right-2 text-[10px] bg-rose-500 text-white px-1 rounded animate-pulse">Trainable</div>
                                 <div className="font-bold text-slate-800">Head</div>
                                 <div className="text-xs text-slate-500">GlobalPooling {'->'} Dense(26)</div>
                                 <div className="text-[10px] font-mono mt-1">(1280, 7, 7) {'->'} (1280) {'->'} (26)</div>
                                 {/* Pulsing indicator */}
                                 <div className="absolute bottom-2 right-2 w-3 h-3 bg-rose-400 rounded-full animate-ping"></div>
                             </div>
                         </div>
                     </div>
                 ) : archLabMode === 'conv' ? (
                     <div className="bg-white p-6 rounded-xl border border-purple-200 shadow-sm max-w-md">
                         <div className="flex justify-between items-center mb-4">
                             <h3 className="text-sm font-bold text-purple-700">卷积操作演示 (Convolution)</h3>
                             <button
                                 onClick={() => setIsConvAnimating(!isConvAnimating)}
                                 className="px-3 py-1 text-xs bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors flex items-center gap-1"
                             >
                                 {isConvAnimating ? <Pause size={12}/> : <Play size={12}/>}
                                 {isConvAnimating ? 'Pause' : 'Animate'}
                             </button>
                         </div>
                         <div className="space-y-4">
                             <div className="relative grid grid-cols-5 gap-1 w-48 h-48 mx-auto bg-slate-50 p-2 rounded overflow-hidden">
                                 {/* 3x3 Kernel Overlay */}
                                 <div 
                                     className="absolute border-2 border-purple-500 bg-purple-500/20 rounded-sm pointer-events-none transition-all duration-500"
                                     style={{
                                         width: 'calc(33.33% - 2px)',
                                         height: 'calc(33.33% - 2px)',
                                         left: isConvAnimating 
                                             ? `${(convKernelPos.col * 33.33)}%` 
                                             : '33.33%',
                                         top: isConvAnimating 
                                             ? `${(convKernelPos.row * 33.33)}%` 
                                             : '33.33%',
                                         boxShadow: '0 0 10px rgba(168, 85, 247, 0.5)',
                                     }}
                                 >
                                     <div className="absolute inset-0 bg-purple-400/30 animate-pulse"></div>
                                 </div>
                                 
                                 {Array.from({length: 25}).map((_, i) => {
                                     const row = Math.floor(i / 5);
                                     const col = i % 5;
                                     const isInKernel = isConvAnimating 
                                         ? row >= convKernelPos.row && row < convKernelPos.row + 3 &&
                                           col >= convKernelPos.col && col < convKernelPos.col + 3
                                         : row >= 1 && row < 4 && col >= 1 && col < 4;
                                     
                                     return (
                                         <div 
                                             key={i} 
                                             className={`border rounded-sm transition-all duration-300 ${
                                                 isInKernel 
                                                     ? 'bg-purple-500 border-purple-600 scale-110 z-10 shadow-lg' 
                                                     : 'bg-white border-slate-200'
                                             }`}
                                         />
                                     );
                                 })}
                             </div>
                             
                             {/* Animation Control */}
                             {isConvAnimating && (
                                 <div className="text-center">
                                     <div className="text-xs text-purple-600 font-mono mb-2">
                                         Position: ({convKernelPos.row}, {convKernelPos.col})
                                     </div>
                                 </div>
                             )}
                             
                             <div className="text-center text-xs text-slate-500">
                                 3x3 Kernel 在 5x5 Feature Map 上滑动
                             </div>
                             <div className="bg-purple-50 p-3 rounded border border-purple-200 text-xs text-purple-700">
                                 <strong>原理：</strong>卷积核在特征图上滑动，每次计算局部区域的加权和，提取局部特征（如边缘、纹理）。
                             </div>
                         </div>
                     </div>
                 ) : (
                     <div className="bg-white p-6 rounded-xl border border-emerald-200 shadow-sm max-w-md">
                         <div className="flex justify-between items-center mb-4">
                             <h3 className="text-sm font-bold text-emerald-700">全局池化 (Global Pooling)</h3>
                             <button
                                 onClick={() => {
                                     setPoolingStep(0);
                                     const interval = setInterval(() => {
                                         setPoolingStep(prev => {
                                             if (prev >= 2) {
                                                 clearInterval(interval);
                                                 return 2;
                                             }
                                             return prev + 1;
                                         });
                                     }, 1000);
                                 }}
                                 className="px-3 py-1 text-xs bg-emerald-600 text-white rounded hover:bg-emerald-700 transition-colors flex items-center gap-1"
                             >
                                 <Play size={12}/> Animate
                             </button>
                         </div>
                         <div className="space-y-4">
                             {/* Before Pooling */}
                             <div className="relative">
                                 <div className={`grid grid-cols-7 gap-1 bg-slate-900 p-2 rounded transition-all duration-1000 ${
                                     poolingStep >= 1 ? 'opacity-50 scale-95' : 'opacity-100'
                                 }`}>
                                     {Array.from({length: 49}).map((_, i) => {
                                         const row = Math.floor(i / 7);
                                         const col = i % 7;
                                         const delay = (row * 7 + col) * 0.02;
                                         return (
                                             <div 
                                                 key={i} 
                                                 className={`aspect-square bg-emerald-400 rounded-sm transition-all duration-500 ${
                                                     poolingStep >= 1 ? 'opacity-30 scale-90' : 'opacity-60'
                                                 }`}
                                                 style={{
                                                     animation: poolingStep === 0 
                                                         ? `pulse 1s ease-in-out ${delay}s infinite` 
                                                         : 'none'
                                                 }}
                                             />
                                         );
                                     })}
                                 </div>
                                 <style>{`
                                     @keyframes pulse {
                                         0%, 100% { opacity: 0.4; }
                                         50% { opacity: 0.8; }
                                     }
                                 `}</style>
                                 <div className="text-[10px] text-slate-500 mt-1 text-center">(1280, 7, 7)</div>
                             </div>
                             
                             {/* Arrow with animation */}
                             <div className="relative flex justify-center">
                                 <ArrowRight className="text-slate-300 rotate-90 transition-all duration-500"/>
                                 {poolingStep >= 1 && (
                                     <div className="absolute inset-0 flex items-center justify-center">
                                         <div className="w-8 h-1 bg-gradient-to-r from-transparent via-emerald-400 to-transparent animate-[flowDown_1s_ease-in-out]"></div>
                                     </div>
                                 )}
                                 <style>{`
                                     @keyframes flowDown {
                                         0% { transform: translateY(-10px); opacity: 0; }
                                         50% { opacity: 1; }
                                         100% { transform: translateY(10px); opacity: 0; }
                                     }
                                 `}</style>
                             </div>
                             
                             {/* After Pooling */}
                             <div className={`bg-emerald-50 p-4 rounded border border-emerald-200 text-center transition-all duration-1000 ${
                                 poolingStep >= 2 ? 'scale-110 shadow-lg' : 'scale-100'
                             }`}>
                                 <div className="text-xs text-emerald-700 font-mono mb-2">(1280,)</div>
                                 <div className="text-[10px] text-emerald-600 mt-1">每个通道取平均值</div>
                                 {poolingStep >= 2 && (
                                     <div className="mt-2 flex justify-center gap-1">
                                         {Array.from({length: 5}).map((_, i) => (
                                             <div 
                                                 key={i}
                                                 className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"
                                                 style={{ animationDelay: `${i * 0.1}s` }}
                                             />
                                         ))}
                                     </div>
                                 )}
                             </div>
                             
                             <div className="bg-emerald-50 p-3 rounded border border-emerald-200 text-xs text-emerald-700">
                                 <strong>作用：</strong>将空间维度 (7x7) 压缩为 1，减少参数量，防止过拟合，同时保留通道信息。
                             </div>
                         </div>
                     </div>
                 )}
            </div>
       </div>

       <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
            <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-cyan-400 sticky top-0">
               <Terminal size={14} /> <span className="text-xs font-bold">Source Code</span>
            </div>
            <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre custom-scrollbar">
{`import mindspore.nn as nn

class MobileNetV2Head(nn.Cell):
    """
    自定义分类头 (Classification Head)
    """
    def __init__(self, input_channel=1280, num_classes=26):
        super(MobileNetV2Head, self).__init__()
        
        # 1. 全局平均池化 (Global Avg Pooling)
        # 将 (Batch, 1280, 7, 7) -> (Batch, 1280)
        # 减少参数量，防止过拟合
        self.flatten = GlobalPooling(reduction='mean')
        
        # 2. 全连接层 (Dense Layer)
        # 将 1280 维特征映射到 26 个分类
        self.dense = nn.Dense(input_channel, num_classes, 
                            weight_init='ones', 
                            has_bias=False)
        
        # 3. 激活函数 (Softmax)
        # 输出概率分布
        self.activation = nn.Softmax()

    def construct(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return self.activation(x)

# 组合模型
backbone = MobileNetV2Backbone()
# 冻结骨干网络参数 (关键步骤)
for param in backbone.get_parameters():
    param.requires_grad = False

head = MobileNetV2Head(input_channel=backbone.out_channels, num_classes=26)
network = mobilenet_v2(backbone, head)
`}
            </div>
       </div>
    </div>
  );

  // Training animation effect
  useEffect(() => {
    let interval: any;
    if (isTraining && epoch < 1000) {
      interval = setInterval(() => {
        setEpoch(prev => {
          const newEpoch = prev + 1;
          const progress = newEpoch / 1000;
          setLoss(0.8 * Math.exp(-4 * progress) + 0.1 + (Math.random() * 0.05));
          return newEpoch >= 1000 ? 1000 : newEpoch;
        });
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isTraining, epoch]);

  // Conv animation effect
  useEffect(() => {
    let interval: any;
    if (isConvAnimating && archLabMode === 'conv') {
      interval = setInterval(() => {
        setConvKernelPos(prev => {
          let newCol = prev.col + 1;
          let newRow = prev.row;
          
          if (newCol > 2) { // 5x5 grid, 3x3 kernel can move 3 positions (0,1,2)
            newCol = 0;
            newRow += 1;
            if (newRow > 2) {
              newRow = 0;
            }
          }
          
          return { row: newRow, col: newCol };
        });
      }, 800);
    }
    return () => clearInterval(interval);
  }, [isConvAnimating, archLabMode]);

  const renderTrain = () => (
    <div className="h-full flex flex-col lg:flex-row animate-fade-in-up overflow-hidden">
       <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
           <div className="p-4 border-b border-cyan-100 bg-cyan-50/30 flex justify-between items-center">
               <h2 className="text-lg font-bold text-cyan-900 flex items-center gap-2">
                   <Zap className="text-cyan-600" /> 3. 训练策略 (Training)
               </h2>
               <div className="flex gap-2">
                   <button 
                       onClick={() => setIsTraining(!isTraining)}
                       className="flex items-center gap-2 px-3 py-1.5 bg-cyan-600 text-white rounded text-xs font-bold hover:bg-cyan-700"
                   >
                       {isTraining ? <Pause size={14}/> : <Play size={14}/>} {isTraining ? 'Pause' : 'Start'}
                   </button>
                   <button 
                       onClick={() => {setIsTraining(false); setEpoch(0); setLoss(0.8);}}
                       className="p-1.5 rounded border border-slate-200 hover:bg-slate-100"
                   >
                       <RotateCcw size={14} className="text-slate-600"/>
                   </button>
               </div>
           </div>
           <div className="flex-1 overflow-y-auto p-4 space-y-6">
               {/* Training Progress */}
               <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                   <div className="flex justify-between items-center mb-4">
                       <h3 className="font-bold text-slate-700 flex items-center gap-2">
                           <Activity size={16} className="text-cyan-500"/> 训练进度
                       </h3>
                       <div className="text-sm font-mono text-slate-600">Epoch: {epoch}/1000</div>
                   </div>
                   <div className="space-y-4">
                       <div>
                           <div className="flex justify-between text-xs text-slate-500 mb-1">
                               <span>Loss</span>
                               <span className="font-mono text-rose-600">{loss.toFixed(4)}</span>
                           </div>
                           <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                               <div 
                                   className="h-full bg-gradient-to-r from-rose-500 to-rose-400 transition-all duration-300"
                                   style={{ width: `${Math.min(100, (1 - loss / 0.8) * 100)}%` }}
                               />
                           </div>
                       </div>
                   </div>
               </div>

               {/* Learning Rate Schedule */}
               <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                   <h3 className="font-bold text-slate-700 mb-4">Learning Rate Schedule</h3>
                   <div className="flex gap-2 mb-4">
                       {['cosine', 'square', 'constant'].map(t => (
                           <button 
                               key={t} 
                               onClick={() => setLrType(t as any)} 
                               className={`px-3 py-1.5 text-xs rounded border font-bold transition-colors ${
                                   lrType===t
                                       ?'bg-cyan-50 border-cyan-200 text-cyan-700'
                                       :'border-slate-200 text-slate-500 hover:bg-slate-50'
                               }`}
                           >
                               {t}
                           </button>
                       ))}
                   </div>
                   <div className="h-40 bg-slate-50 border border-slate-100 rounded relative overflow-hidden">
                       <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none" viewBox="0 0 300 100">
                           {lrType === 'cosine' && (
                               <path 
                                   d="M 0 100 Q 75 50 150 0 Q 225 50 300 100" 
                                   stroke="#0ea5e9" 
                                   strokeWidth="2" 
                                   fill="none"
                               />
                           )}
                           {lrType === 'square' && (
                               <path 
                                   d="M 0 100 L 300 0" 
                                   stroke="#0ea5e9" 
                                   strokeWidth="2" 
                                   fill="none"
                               />
                           )}
                           {lrType === 'constant' && (
                               <line 
                                   x1="0" y1="50" x2="300" y2="50" 
                                   stroke="#0ea5e9" 
                                   strokeWidth="2"
                               />
                           )}
                       </svg>
                       <div className="absolute bottom-2 left-2 text-[10px] text-slate-400">LR: 0.001</div>
                       <div className="absolute bottom-2 right-2 text-[10px] text-slate-400">LR: {lrType === 'constant' ? '0.001' : '0.0'}</div>
                   </div>
                   <div className="text-xs text-slate-500 mt-2 text-center">Epochs (0 {'->'} 1000)</div>
               </div>

               {/* Transfer Learning Concept */}
               <div className="bg-cyan-50 p-4 rounded-xl border border-cyan-200">
                   <h4 className="font-bold text-cyan-800 mb-2 text-sm flex items-center gap-2">
                       <BookOpen size={14}/> 迁移学习策略
                   </h4>
                   <div className="text-xs text-cyan-700 space-y-2">
                       <p>1. <strong>冻结 Backbone：</strong>保持 ImageNet 预训练权重不变，只训练 Head 部分</p>
                       <p>2. <strong>优势：</strong>利用通用视觉特征，快速适应新任务，减少训练时间</p>
                       <p>3. <strong>特征提取：</strong>先提取所有样本的特征图，保存为 .npy 文件，训练时直接加载</p>
                   </div>
               </div>
           </div>
       </div>

       <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
            <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-cyan-400 sticky top-0">
               <Terminal size={14} /> <span className="text-xs font-bold">Source Code</span>
            </div>
            <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre custom-scrollbar">
{`# 1. 特征提取 (Feature Extraction)
def extract_features(net, dataset_path, config):
    """
    提取并保存特征图，避免重复计算
    """
    if not os.path.exists(config.features_path):
        os.makedirs(config.features_path)
    
    dataset = create_dataset(config=config)
    step_size = dataset.get_dataset_size()
    data_iter = dataset.create_dict_iterator()
    
    for i, data in enumerate(data_iter):
        features_path = os.path.join(
            config.features_path, f"feature_{i}.npy"
        )
        label_path = os.path.join(
            config.features_path, f"label_{i}.npy"
        )
        
        if not os.path.exists(features_path):
            image = data["image"]
            label = data["label"]
            # 只通过 Backbone，不经过 Head
            features = net(image)
            np.save(features_path, features.asnumpy())
            np.save(label_path, label.asnumpy())
        
        print(f"Complete batch {i+1}/{step_size}")

# 2. 学习率调度器
import math

def build_lr(total_steps, lr_max=0.001, decay_type='cosine'):
    """
    生成学习率数组
    """
    lr_each_step = []
    for i in range(total_steps):
        if decay_type == 'cosine':
            # 余弦退火：平滑下降
            decay = 0.5 * (1 + math.cos(math.pi * i / total_steps))
            lr = lr_max * decay
        elif decay_type == 'square':
            # 平方衰减：快速下降
            frac = 1.0 - i / total_steps
            lr = lr_max * (frac * frac)
        else:
            lr = lr_max
            
        lr_each_step.append(lr)
        
    return lr_each_step

# 3. 训练循环
def train_head():
    train_dataset = create_dataset(config=config)
    step_size = train_dataset.get_dataset_size()
    
    # 冻结 Backbone
    backbone = MobileNetV2Backbone()
    for param in backbone.get_parameters():
        param.requires_grad = False
    load_checkpoint(config.pretrained_ckpt, net=backbone)
    
    # 只训练 Head
    head = MobileNetV2Head(
        input_channel=backbone.out_channels, 
        num_classes=config.num_classes
    )
    network = mobilenet_v2(backbone, head)
    
    # 损失函数和优化器
    loss = nn.SoftmaxCrossEntropyWithLogits(
        sparse=True, reduction='mean'
    )
    lrs = build_lr(
        config.epochs * step_size, 
        lr_max=config.lr_max, 
        decay_type=config.decay_type
    )
    opt = nn.Momentum(
        head.trainable_params(), 
        lrs, 
        config.momentum, 
        config.weight_decay
    )
    
    # 训练
    net = nn.WithLossCell(head, loss)
    train_step = nn.TrainOneStepCell(net, opt)
    train_step.set_train()
    
    history = []
    features_path = config.features_path
    
    for epoch in range(config.epochs):
        idx_list = list(range(step_size))
        random.shuffle(idx_list)
        losses = []
        
        for j in idx_list:
            # 加载预提取的特征
            feature = Tensor(
                np.load(os.path.join(features_path, f"feature_{j}.npy"))
            )
            label = Tensor(
                np.load(os.path.join(features_path, f"label_{j}.npy"))
            )
            losses.append(train_step(feature, label).asnumpy())
        
        epoch_loss = np.mean(np.array(losses))
        history.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
`}
            </div>
       </div>
    </div>
  );

  const renderInference = () => (
    <div className="h-full flex flex-col lg:flex-row animate-fade-in-up overflow-hidden">
       <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
           <div className="p-4 border-b border-cyan-100 bg-cyan-50/30">
               <h2 className="text-lg font-bold text-cyan-900 flex items-center gap-2">
                   <Search className="text-cyan-600" /> 4. 推理 (Inference)
               </h2>
           </div>
           <div className="flex-1 overflow-y-auto p-4 flex flex-col items-center justify-center">
               <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 flex flex-col items-center gap-6">
                   <div className="flex items-center gap-4">
                       <div className="w-16 h-16 bg-amber-100 rounded flex items-center justify-center text-2xl border border-amber-200">🧢</div>
                       <ArrowRight className="text-slate-300"/>
                       <div className={`w-16 h-16 bg-slate-100 rounded flex items-center justify-center border transition-all ${inferState===1?'border-blue-500 shadow shadow-blue-200':''}`}>
                           {inferState===1?<Loader2 className="animate-spin text-blue-500"/>:<Cpu className="text-slate-400"/>}
                       </div>
                       <ArrowRight className="text-slate-300"/>
                       <div className="font-mono font-bold text-lg text-emerald-600">
                           {inferState===2 ? "Hats" : "?"}
                       </div>
                   </div>
                   <button onClick={()=>{setInferState(1);setTimeout(()=>setInferState(2),1000)}} className="px-6 py-2 bg-cyan-600 text-white rounded font-bold shadow hover:bg-cyan-700 text-sm">Predict</button>
               </div>
           </div>
       </div>

       <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
            <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-cyan-400 sticky top-0">
               <Terminal size={14} /> <span className="text-xs font-bold">Source Code</span>
            </div>
            <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre custom-scrollbar">
{`def infer_one(network, image_path):
    """
    单张图片推理
    """
    # 1. 读取并预处理
    image = Image.open(image_path)
    # 调用之前的流水线: Resize -> Norm -> HWC2CHW
    tensor = image_process(image)
    
    # 2. 模型前向计算
    # logits shape: (1, 26)
    logits = network(tensor)
    
    # 3. 获取最大概率索引 (Argmax)
    # asnumpy() 将 Tensor 转为 Numpy 数组
    pred_idx = np.argmax(logits.asnumpy(), axis=1)[0]
    
    # 4. 查字典获取类别名
    # inverted = {0: 'Bottle', 1: 'Hats', ...}
    class_name = inverted[pred_idx]
    
    print(f"File: {image_path}, Pred: {class_name}")
    return pred_idx

# 加载训练好的 Checkpoint
param_dict = load_checkpoint("mobilenetv2-500.ckpt")
load_param_into_net(network, param_dict)
network.set_train(False) # 设为评估模式
`}
            </div>
       </div>
    </div>
  );

  switch (activeSubTab) {
    case 'CV_INTRO': return renderIntro();
    case 'CV_PIPELINE': return renderPipeline();
    case 'CV_ARCH': return renderArch();
    case 'CV_TRAIN': return renderTrain();
    case 'CV_INFERENCE': return renderInference();
    default: return renderIntro();
  }
};