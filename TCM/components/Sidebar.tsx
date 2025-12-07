import React from 'react';
import { 
  LayoutDashboard, 
  BrainCircuit, 
  Layers, 
  Ruler, 
  Brain, 
  ChevronRight 
} from 'lucide-react';
import { StepType } from '../types';

interface SidebarProps {
  currentStep: StepType;
  onStepChange: (step: StepType) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentStep, onStepChange }) => {
  const menuItems = [
    { id: StepType.OVERVIEW, icon: LayoutDashboard, label: '0. 核心流程视窗', subLabel: 'Pipeline Overview' },
    { id: StepType.LLM_INFERENCE, icon: BrainCircuit, label: '1. Prompt 策略模拟', subLabel: 'LLM Inference' },
    { id: StepType.BATCH_PROCESSING, icon: Layers, label: '2. 批量处理逻辑', subLabel: 'Batch Processing' },
    { id: StepType.EMBEDDING_METRICS, icon: Ruler, label: '3. 核心概念百科', subLabel: 'Scoring Metrics' },
  ];

  return (
    <div className="w-72 bg-white border-r border-slate-200 flex flex-col shadow-lg z-20 flex-shrink-0">
      {/* Sidebar Header */}
      <div className="p-6 border-b border-slate-100">
           <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-purple-600 rounded-xl flex items-center justify-center text-white shadow-purple-200 shadow-md">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                  <h1 className="font-bold text-lg text-slate-800 tracking-tight leading-none">中医辨证 Lab</h1>
                  <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">TCM Experiment</span>
              </div>
           </div>
      </div>
      
      {/* Navigation List */}
      <nav className="flex-1 overflow-y-auto py-6 px-4 space-y-2">
        {menuItems.map((item) => {
          const isActive = currentStep === item.id;
          const Icon = item.icon;
          
          return (
            <button
              key={item.id}
              onClick={() => onStepChange(item.id)}
              className={`w-full text-left flex items-center gap-4 px-4 py-3.5 rounded-xl transition-all duration-200 group relative
                ${isActive 
                  ? 'bg-purple-600 text-white shadow-md shadow-purple-200' 
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900 border border-transparent hover:border-slate-100'
                }
              `}
            >
              <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-purple-100' : 'text-slate-400 group-hover:text-purple-500'}`} />
              
              <div className="flex-1 min-w-0">
                  <div className={`font-medium text-sm truncate ${isActive ? 'text-white' : 'text-slate-700'}`}>
                      {item.label}
                  </div>
                  <div className={`text-[11px] truncate mt-0.5 ${isActive ? 'text-purple-200' : 'text-slate-400 group-hover:text-slate-500'}`}>
                      {item.subLabel}
                  </div>
              </div>

              {isActive && (
                  <ChevronRight className="w-4 h-4 text-purple-300" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Sidebar Footer */}
      <div className="p-4 border-t border-slate-100 bg-slate-50/50">
           <div className="flex items-center gap-2 p-3 bg-white border border-slate-200 rounded-lg">
              <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
              <span className="text-xs font-mono text-slate-500">Python 3.9 Ready</span>
           </div>
      </div>
    </div>
  );
};
