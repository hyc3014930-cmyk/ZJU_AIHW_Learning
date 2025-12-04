import React, { useState } from 'react';
import { Database, Brain, Activity, Search, Code, Layers, BookOpen, MessageSquare, Scale, Image as ImageIcon, Box, LayoutGrid, Zap, ScanLine, ListVideo, Calculator } from 'lucide-react';
import { TabView, Project } from './types';
import { DataView } from './components/DataView';
import { ModelView } from './components/ModelView';
import { TrainingView } from './components/TrainingView';
import { InferenceView } from './components/InferenceView';
import { IntroView } from './components/IntroView';
import { TcmView } from './components/TcmView';
import { VisionView } from './components/VisionView';
import StartPage from './components/StartPage';

const App: React.FC = () => {
  const [currentProject, setCurrentProject] = useState<Project>(Project.DGRAPH);
  const [activeTab, setActiveTab] = useState<string>(TabView.INTRO);
    const [showStart, setShowStart] = useState<boolean>(true);

  // Switch project handler
  const handleProjectSwitch = (p: Project) => {
    setCurrentProject(p);
    // Set default tab for each project
    if (p === Project.DGRAPH) setActiveTab(TabView.INTRO);
    if (p === Project.TCM) setActiveTab(TabView.TCM_INTRO);
    if (p === Project.VISION) setActiveTab(TabView.CV_INTRO);
  };

  const renderContent = () => {
        if (showStart) {
            return (
                <StartPage onSelect={(m) => {
                    // module mapping:
                    // 1 -> 金融异常检测 (DGraph project)
                    // 2 -> TCM 中医大模型助手 (TCM project)
                    // 3 -> Garbage 垃圾分类视觉模型 (Vision project)
                    setShowStart(false);
                    if (m === 1) {
                        setCurrentProject(Project.DGRAPH);
                        setActiveTab(TabView.INTRO);
                    }
                    if (m === 2) {
                        setCurrentProject(Project.TCM);
                        setActiveTab(TabView.TCM_INTRO);
                    }
                    if (m === 3) {
                        setCurrentProject(Project.VISION);
                        setActiveTab(TabView.CV_INTRO);
                    }
                }} onCancel={() => setShowStart(false)} />
            );
        }
    // 1. DGraph (Finance)
    if (currentProject === Project.DGRAPH) {
        switch (activeTab) {
        case TabView.INTRO: return <IntroView />;
        case TabView.DATA: return <DataView />;
        case TabView.MODEL: return <ModelView />;
        case TabView.TRAINING: return <TrainingView />;
        case TabView.INFERENCE: return <InferenceView />;
        default: return <IntroView />;
        }
    }
    // 2. TCM (LLM)
    if (currentProject === Project.TCM) {
        return <TcmView activeSubTab={activeTab} />;
    }
    // 3. Vision (CV)
    if (currentProject === Project.VISION) {
        return <VisionView activeSubTab={activeTab} />;
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 flex font-sans overflow-hidden">
      
      {/* 1. Global Project Sidebar (Leftmost) */}
      <aside className="w-20 bg-slate-900 flex flex-col items-center py-6 gap-6 z-20 shadow-xl flex-shrink-0">
         <div className="bg-blue-600 p-3 rounded-xl text-white shadow-lg shadow-blue-900/50 mb-4">
             <Code size={24} />
         </div>
         
         <ProjectIcon 
            active={currentProject === Project.DGRAPH} 
            onClick={() => handleProjectSwitch(Project.DGRAPH)}
            icon={<LayoutGrid size={24} />}
            label="Finance"
            color="text-blue-400"
         />
         <ProjectIcon 
            active={currentProject === Project.TCM} 
            onClick={() => handleProjectSwitch(Project.TCM)}
            icon={<MessageSquare size={24} />}
            label="TCM-LLM"
            color="text-emerald-400"
         />
         <ProjectIcon 
            active={currentProject === Project.VISION} 
            onClick={() => handleProjectSwitch(Project.VISION)}
            icon={<ImageIcon size={24} />}
            label="Vision"
            color="text-cyan-400"
         />
      </aside>

      {/* 2. Project Navigation Sidebar & Main Content */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        
        {/* Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 flex-shrink-0">
            <div>
                <h1 className="text-lg font-bold text-slate-800 tracking-tight flex items-center gap-2">
                    {currentProject === Project.DGRAPH && <><LayoutGrid size={20} className="text-blue-600"/> DGraphFin 金融风控模型</>}
                    {currentProject === Project.TCM && <><Brain size={20} className="text-emerald-600"/> TCM 中医大模型助手</>}
                    {currentProject === Project.VISION && <><Box size={20} className="text-cyan-600"/> Garbage 垃圾分类视觉模型</>}
                </h1>
            </div>
            <div className="flex items-center gap-2 text-xs font-bold text-slate-500 bg-slate-100 px-3 py-1.5 rounded-full">
                <Activity size={14} />
                <span>AI Teaching Platform v2.1</span>
            </div>
        </header>

        {/* Workspace */}
        <div className="flex-1 flex overflow-hidden">
            
            {/* Sub-Navigation Sidebar */}
            <nav className="w-64 bg-white border-r border-slate-200 p-4 flex flex-col gap-2 overflow-y-auto flex-shrink-0 hidden md:flex">
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 px-2">Menu</div>
                
                {currentProject === Project.DGRAPH && (
                    <>
                        <NavButton active={activeTab === TabView.INTRO} onClick={() => setActiveTab(TabView.INTRO)} icon={<BookOpen size={18} />} label="0. 项目背景" />
                        <NavButton active={activeTab === TabView.DATA} onClick={() => setActiveTab(TabView.DATA)} icon={<Database size={18} />} label="1. 数据准备" />
                        <NavButton active={activeTab === TabView.MODEL} onClick={() => setActiveTab(TabView.MODEL)} icon={<Layers size={18} />} label="2. MLP 模型架构" />
                        <NavButton active={activeTab === TabView.TRAINING} onClick={() => setActiveTab(TabView.TRAINING)} icon={<Activity size={18} />} label="3. 训练循环" />
                        <NavButton active={activeTab === TabView.INFERENCE} onClick={() => setActiveTab(TabView.INFERENCE)} icon={<Search size={18} />} label="4. 预测推理" />
                    </>
                )}

                {currentProject === Project.TCM && (
                    <>
                        <NavButton active={activeTab === TabView.TCM_INTRO} onClick={() => setActiveTab(TabView.TCM_INTRO)} icon={<BookOpen size={18} />} label="0. 项目背景" />
                        <NavButton active={activeTab === TabView.TCM_PROMPT} onClick={() => setActiveTab(TabView.TCM_PROMPT)} icon={<MessageSquare size={18} />} label="1. 提示工程 (Prompt)" />
                        <NavButton active={activeTab === TabView.TCM_BATCH} onClick={() => setActiveTab(TabView.TCM_BATCH)} icon={<ListVideo size={18} />} label="2. 批量处理 (Batch)" />
                        <NavButton active={activeTab === TabView.TCM_EVAL} onClick={() => setActiveTab(TabView.TCM_EVAL)} icon={<Calculator size={18} />} label="3. 效果评估 (Score)" />
                    </>
                )}

                {currentProject === Project.VISION && (
                    <>
                        <NavButton active={activeTab === TabView.CV_INTRO} onClick={() => setActiveTab(TabView.CV_INTRO)} icon={<BookOpen size={18} />} label="0. 项目背景" />
                        <NavButton active={activeTab === TabView.CV_PIPELINE} onClick={() => setActiveTab(TabView.CV_PIPELINE)} icon={<ScanLine size={18} />} label="1. 图像流水线" />
                        <NavButton active={activeTab === TabView.CV_ARCH} onClick={() => setActiveTab(TabView.CV_ARCH)} icon={<Layers size={18} />} label="2. MobileNet 架构" />
                        <NavButton active={activeTab === TabView.CV_TRAIN} onClick={() => setActiveTab(TabView.CV_TRAIN)} icon={<Zap size={18} />} label="3. 迁移训练 & LR" />
                        <NavButton active={activeTab === TabView.CV_INFERENCE} onClick={() => setActiveTab(TabView.CV_INFERENCE)} icon={<Search size={18} />} label="4. 预测推理" />
                    </>
                )}
            </nav>

            {/* Viewport */}
            <main className="flex-1 bg-slate-50 relative overflow-hidden flex flex-col">
                <div className="flex-1 overflow-hidden">
                    {renderContent()}
                </div>
            </main>

        </div>
      </div>
    </div>
  );
};

// --- Sub Components ---

const ProjectIcon: React.FC<{ active: boolean, onClick: () => void, icon: React.ReactNode, label: string, color: string }> = ({ active, onClick, icon, label, color }) => (
    <button 
       onClick={onClick}
       className={`w-12 h-12 rounded-2xl flex flex-col items-center justify-center gap-1 transition-all duration-300 group relative ${active ? 'bg-slate-800' : 'hover:bg-slate-800/50'}`}
    >
        <div className={`transition-colors ${active ? color : 'text-slate-500 group-hover:text-slate-300'}`}>
            {icon}
        </div>
        <span className={`text-[9px] font-bold ${active ? 'text-white' : 'text-slate-500'}`}>{label}</span>
        {active && <div className={`absolute left-0 w-1 h-8 rounded-r-full ${color.replace('text', 'bg')}`}></div>}
    </button>
);

const NavButton: React.FC<{ active: boolean, onClick: () => void, icon: React.ReactNode, label: string }> = ({ active, onClick, icon, label }) => (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-3 rounded-lg flex items-center gap-3 transition-all ${
        active 
          ? 'bg-slate-100 text-slate-900 font-bold shadow-sm' 
          : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
      }`}
    >
      <div className={active ? 'text-blue-600' : 'text-slate-400'}>{icon}</div>
      <span className="text-sm">{label}</span>
    </button>
);

export default App;