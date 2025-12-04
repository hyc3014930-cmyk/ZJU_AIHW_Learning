import React from 'react';

const StartPage: React.FC<{ onSelect: (m: number) => void; onCancel?: () => void }> = ({ onSelect, onCancel }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-6">
      <div className="max-w-5xl w-full">
        <div className="bg-white p-8 rounded-2xl shadow-2xl border border-slate-200">
          <h1 className="text-2xl font-bold text-slate-800 mb-4">请选择一个作业模块</h1>
          <p className="text-sm text-slate-500 mb-6">下面列出三个模块，点击进入相应实验与交互演示（中文命名）。</p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button onClick={() => onSelect(1)} className="text-left bg-cyan-50 border border-cyan-100 rounded-xl p-4 hover:shadow-md transition">
              <div className="text-sm text-cyan-700 font-bold">金融异常检测</div>
              <div className="text-xs text-slate-500 mt-2">面向金融场景的异常检测任务与可视化解释。</div>
            </button>

            <button onClick={() => onSelect(2)} className="text-left bg-emerald-50 border border-emerald-100 rounded-xl p-4 hover:shadow-md transition">
              <div className="text-sm text-emerald-700 font-bold">TCM 中医大模型助手</div>
              <div className="text-xs text-slate-500 mt-2">中医相关的提示工程、批量处理与评估实验。</div>
            </button>

            <button onClick={() => onSelect(3)} className="text-left bg-rose-50 border border-rose-100 rounded-xl p-4 hover:shadow-md transition">
              <div className="text-sm text-rose-700 font-bold">Garbage 垃圾分类视觉模型</div>
              <div className="text-xs text-slate-500 mt-2">结合视觉模型和代码生成动画，解释垃圾分类流程。</div>
            </button>
          </div>

          <div className="mt-6 flex justify-end gap-2">
            <button onClick={() => onCancel && onCancel()} className="px-4 py-2 rounded bg-slate-100 text-slate-700 text-sm">跳过</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StartPage;
