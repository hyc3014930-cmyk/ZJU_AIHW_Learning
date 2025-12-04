
import React, { useState } from 'react';
import { Image as ImageIcon, Layers, Cpu, ArrowRight, Settings, Box, BarChart3, ScanLine, Terminal, BookOpen, Trash2, Zap, Search, Loader2, History, Scaling, Grid, Minimize } from 'lucide-react';

export const VisionView: React.FC<{ activeSubTab: string }> = ({ activeSubTab }) => {
  const [pipelineStep, setPipelineStep] = useState(0); 
  const [lrType, setLrType] = useState<'cosine' | 'square' | 'constant'>('cosine');
  const [inferState, setInferState] = useState(0);
  const [archLabMode, setArchLabMode] = useState<'struct' | 'conv' | 'pool'>('struct'); 

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
                 </div>

                 {archLabMode === 'struct' ? (
                     <div className="space-y-4 w-full max-w-md">
                         <div className="bg-slate-200 p-4 rounded-lg border border-slate-300 opacity-70 relative">
                             <div className="absolute top-2 right-2 text-[10px] bg-slate-400 text-white px-1 rounded">Frozen</div>
                             <div className="font-bold text-slate-700">Backbone</div>
                             <div className="text-xs text-slate-500">MobileNetV2 (ImageNet)</div>
                             <div className="text-[10px] font-mono mt-1">Out: 1280 channels</div>
                         </div>
                         <div className="flex justify-center"><ArrowRight className="text-slate-300 rotate-90"/></div>
                         <div className="bg-white p-4 rounded-lg border-2 border-rose-400 relative shadow-lg">
                             <div className="absolute top-2 right-2 text-[10px] bg-rose-500 text-white px-1 rounded animate-pulse">Trainable</div>
                             <div className="font-bold text-slate-800">Head</div>
                             <div className="text-xs text-slate-500">GlobalPooling -> Dense(26)</div>
                         </div>
                     </div>
                 ) : (
                     <div className="bg-white p-4 rounded-xl border border-purple-200 shadow-sm">
                         <h3 className="text-sm font-bold text-purple-700 mb-2">Convolution Demo</h3>
                         <div className="grid grid-cols-5 gap-1 w-40 h-40 mx-auto">
                             {[...Array(25)].map((_,i) => <div key={i} className={`border ${i===12?'bg-purple-500':'bg-slate-50'}`}></div>)}
                         </div>
                         <div className="text-center text-xs text-slate-400 mt-2">3x3 Kernel sliding...</div>
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

  const renderTrain = () => (
    <div className="h-full flex flex-col lg:flex-row animate-fade-in-up overflow-hidden">
       <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
           <div className="p-4 border-b border-cyan-100 bg-cyan-50/30">
               <h2 className="text-lg font-bold text-cyan-900 flex items-center gap-2">
                   <Zap className="text-cyan-600" /> 3. 训练策略 (Training)
               </h2>
           </div>
           <div className="flex-1 overflow-y-auto p-4">
               <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200">
                   <h3 className="font-bold text-slate-700 mb-2">Learning Rate Schedule</h3>
                   <div className="flex gap-2 mb-4">
                       {['cosine', 'square', 'constant'].map(t => (
                           <button key={t} onClick={() => setLrType(t as any)} className={`px-2 py-1 text-xs rounded border ${lrType===t?'bg-blue-50 border-blue-200 text-blue-700':'border-transparent text-slate-500'}`}>{t}</button>
                       ))}
                   </div>
                   <div className="h-32 bg-slate-50 border border-slate-100 rounded relative">
                       <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
                            <path d={`M 0 100 Q 150 ${lrType==='cosine'?100:0} 300 100`} stroke="#0ea5e9" strokeWidth="2" fill="none"/>
                            {/* Simplified visualization */}
                       </svg>
                   </div>
                   <div className="text-xs text-slate-500 mt-2 text-center">Epochs (0 -> 1000)</div>
               </div>
           </div>
       </div>

       <div className="h-[40vh] lg:h-auto lg:w-[450px] bg-slate-900 border-t lg:border-t-0 lg:border-l border-slate-700 flex flex-col flex-shrink-0 z-10 shadow-xl">
            <div className="p-3 border-b border-slate-800 bg-slate-900 flex items-center gap-2 text-cyan-400 sticky top-0">
               <Terminal size={14} /> <span className="text-xs font-bold">Source Code</span>
            </div>
            <div className="flex-1 overflow-x-auto overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300 whitespace-pre custom-scrollbar">
{`import math

def build_lr(total_steps, lr_max=0.001, decay_type='cosine'):
    """
    生成学习率数组
    """
    lr_each_step = []
    for i in range(total_steps):
        if decay_type == 'cosine':
            # 余弦退火：先慢后快再慢
            decay = 0.5 * (1 + math.cos(math.pi * i / total_steps))
            lr = lr_max * decay
        elif decay_type == 'square':
            # 平方衰减
            frac = 1.0 - i / total_steps
            lr = lr_max * (frac * frac)
        else:
            lr = lr_max
            
        lr_each_step.append(lr)
        
    return lr_each_step

# 训练配置
config = {
    "epochs": 1000,
    "lr_max": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.001
}

# 定义优化器
# 只优化 Head 部分的参数 (transfer learning)
opt = nn.Momentum(
    params=head.trainable_params(), 
    learning_rate=build_lr(1000), 
    momentum=config.momentum
)
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
