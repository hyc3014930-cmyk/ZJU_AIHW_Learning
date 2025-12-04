import React from 'react';
import CodeHighlighter from './CodeHighlighter';

const exampleCode = [
  '// 简单实验：数值归一化',
  'function normalize(arr) {',
  '  const max = Math.max(...arr);',
  '  const min = Math.min(...arr);',
  "  return arr.map(x => (x - min) / (max - min));",
  '}',
  '',
  "console.log(normalize([10, 20, 30]));",
].join('\n');

const LabView: React.FC<{ title?: string }> = ({ title = '原理实验室' }) => {
  return (
    <div className="p-4">
      <div className="bg-white p-4 rounded-xl border border-slate-200">
        <h3 className="font-bold text-slate-800 mb-2">{title}</h3>
        <div className="text-sm text-slate-600 mb-4">通过一个小实验动画解释数据归一化原理，并用代码高亮对应步骤。</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex flex-col gap-2">
            <div className="p-4 bg-slate-50 rounded">动画占位：可在此处加入 SVG/Canvas 动画展示数值变换。</div>
          </div>

          <div>
            <CodeHighlighter code={exampleCode} highlightedLines={[2,3,4]} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default LabView;
