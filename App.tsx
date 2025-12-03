import React, { useState } from 'react';
import { Database, Brain, Activity, Search, Code, Layers, BookOpen } from 'lucide-react';
import { TabView } from './types';
import { DataView } from './components/DataView';
import { ModelView } from './components/ModelView';
import { TrainingView } from './components/TrainingView';
import { InferenceView } from './components/InferenceView';
import { IntroView } from './components/IntroView';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabView>(TabView.INTRO);

  const renderContent = () => {
    switch (activeTab) {
      case TabView.INTRO:
        return <IntroView />;
      case TabView.DATA:
        return <DataView />;
      case TabView.MODEL:
        return <ModelView />;
      case TabView.TRAINING:
        return <TrainingView />;
      case TabView.INFERENCE:
        return <InferenceView />;
      default:
        return <IntroView />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg text-white shadow-lg shadow-blue-200">
              <Code size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-800">
                DGraphFin 代码可视化助手
              </h1>
              <div className="text-xs text-slate-500">
                基于 MLP (多层感知机) 的金融欺诈检测
              </div>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-2 text-sm text-slate-500 bg-slate-100 px-3 py-1 rounded-full">
            <Brain size={14} />
            <span>AI 零基础模式</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          
          {/* Sidebar Navigation */}
          <nav className="lg:w-72 flex-shrink-0 space-y-3">
             <NavButton 
              active={activeTab === TabView.INTRO} 
              onClick={() => setActiveTab(TabView.INTRO)}
              icon={<BookOpen size={20} />}
              label="0. 项目背景 (Context)"
              description="金融反欺诈与 DGraph 数据集"
              colorClass="slate"
            />
            <NavButton 
              active={activeTab === TabView.DATA} 
              onClick={() => setActiveTab(TabView.DATA)}
              icon={<Database size={20} />}
              label="1. 数据准备 (Data)"
              description="加载数据、归一化与分组"
              colorClass="blue"
            />
            <NavButton 
              active={activeTab === TabView.MODEL} 
              onClick={() => setActiveTab(TabView.MODEL)}
              icon={<Layers size={20} />}
              label="2. 模型架构 (Model)"
              description="搭建 MLP 神经网络的大脑"
              colorClass="purple"
            />
            <NavButton 
              active={activeTab === TabView.TRAINING} 
              onClick={() => setActiveTab(TabView.TRAINING)}
              icon={<Activity size={20} />}
              label="3. 训练循环 (Train)"
              description="通过 200 轮练习优化模型"
              colorClass="green"
            />
            <NavButton 
              active={activeTab === TabView.INFERENCE} 
              onClick={() => setActiveTab(TabView.INFERENCE)}
              icon={<Search size={20} />}
              label="4. 预测推理 (Predict)"
              description="判断用户是正常还是欺诈"
              colorClass="amber"
            />
            
            <div className="mt-8 p-4 bg-blue-50 rounded-xl border border-blue-100">
              <h4 className="text-blue-800 font-bold text-sm mb-2">💡 什么是 MLP?</h4>
              <p className="text-xs text-blue-700 leading-relaxed">
                全称 Multi-Layer Perceptron (多层感知机)。把它想象成一个由很多层“神经元”组成的筛选器。数据进去，经过层层加权计算，最后输出分类结果。
              </p>
            </div>
          </nav>

          {/* Interactive Viewport */}
          <div className="flex-1 bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-200 overflow-hidden min-h-[600px] flex flex-col">
            {renderContent()}
          </div>
        </div>
      </main>
    </div>
  );
};

interface NavButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  description: string;
  colorClass: 'blue' | 'purple' | 'green' | 'amber' | 'slate';
}

const NavButton: React.FC<NavButtonProps> = ({ active, onClick, icon, label, description, colorClass }) => {
  const activeStyles = {
    blue: 'bg-blue-600 shadow-blue-200',
    purple: 'bg-purple-600 shadow-purple-200',
    green: 'bg-emerald-600 shadow-emerald-200',
    amber: 'bg-amber-600 shadow-amber-200',
    slate: 'bg-slate-800 shadow-slate-300',
  };

  const textActiveStyles = {
    blue: 'text-blue-100',
    purple: 'text-purple-100',
    green: 'text-emerald-100',
    amber: 'text-amber-100',
    slate: 'text-slate-200',
  };

  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-5 py-4 rounded-xl transition-all duration-300 flex items-center gap-4 group relative overflow-hidden ${
        active 
          ? `${activeStyles[colorClass]} text-white shadow-lg scale-[1.02]` 
          : 'bg-white hover:bg-slate-50 text-slate-600 border border-slate-200 hover:border-slate-300'
      }`}
    >
      <div className={`relative z-10 transition-transform duration-300 ${active ? 'scale-110' : 'group-hover:scale-110'}`}>
        {icon}
      </div>
      <div className="relative z-10">
        <div className={`font-bold text-sm mb-0.5 ${active ? 'text-white' : 'text-slate-800'}`}>{label}</div>
        <div className={`text-xs ${active ? textActiveStyles[colorClass] : 'text-slate-500'}`}>{description}</div>
      </div>
    </button>
  );
};

export default App;