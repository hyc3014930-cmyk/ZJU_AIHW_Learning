import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodeViewerProps {
  code: string;
  language?: string;
  highlightLines?: number[];
  title?: string;
}

export const CodeViewer: React.FC<CodeViewerProps> = ({ 
  code, 
  language = 'python', 
  title = 'Source Code' 
}) => {
  return (
    <div className="rounded-lg overflow-hidden border border-slate-700 bg-[#1e1e1e] shadow-2xl flex flex-col h-full">
      <div className="bg-[#2d2d2d] px-4 py-2 flex items-center justify-between border-b border-black/50">
        <div className="flex space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
          <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
          <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
        </div>
        <span className="text-xs text-gray-400 font-mono">{title}</span>
      </div>
      <div className="flex-1 overflow-auto custom-scrollbar">
        <SyntaxHighlighter
          language={language}
          style={vscDarkPlus}
          customStyle={{
            margin: 0,
            padding: '1.5rem',
            background: 'transparent',
            fontSize: '0.9rem',
            lineHeight: '1.5',
            fontFamily: '"JetBrains Mono", monospace'
          }}
          showLineNumbers={true}
          lineNumberStyle={{ minWidth: '2.5em', paddingRight: '1em', color: '#6e7681', textAlign: 'right' }}
          wrapLines={true}
          lineProps={(lineNumber) => {
            const style: React.CSSProperties = { display: 'block' };
            // Simple logic: if specific keywords appear, highlight slightly (mocking intelligent highlighting)
            // In a real app, we'd pass explicit highlight ranges props
            return { style };
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};
