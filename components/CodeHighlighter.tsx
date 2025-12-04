import React from 'react';

const CodeHighlighter: React.FC<{
  code: string;
  highlightedLines?: number[];
  // map of line number (1-based) to tailwind color key, e.g. 'indigo'|'purple'|'emerald'|'rose'|'cyan'
  highlightedMap?: Record<number, string>;
  lineTooltips?: Record<number, string>;
  dark?: boolean;
}> = ({ code, highlightedLines = [], highlightedMap = {}, lineTooltips = {}, dark = false }) => {
  const lines = code.split('\n');
  const set = new Set(highlightedLines.map(n => n - 1));

  const baseClass = dark
    ? 'font-mono text-xs text-slate-300 bg-slate-900 rounded p-3 overflow-x-auto border border-slate-700 relative'
    : 'font-mono text-xs text-slate-700 bg-white rounded p-3 overflow-x-auto border border-slate-100 relative';

  const getHighlightClass = (lineNumber: number) => {
    // lineNumber is 1-based
    const color = highlightedMap[lineNumber];
    if (!color) return set.has(lineNumber - 1)
      ? (dark ? 'bg-yellow-900/30 border-l-4 border-yellow-600' : 'bg-yellow-100/60 border-l-4 border-yellow-300')
      : '';

    // support several semantic colors
    const lightBg = `bg-${color}-100/60 border-l-4 border-${color}-300`;
    const darkBg = `bg-${color}-900/30 border-l-4 border-${color}-600`;
    return dark ? darkBg : lightBg;
  };

  return (
    <pre className={baseClass}>
      {lines.map((ln, i) => {
        const lineNumber = i + 1;
        const hlClass = getHighlightClass(lineNumber);
        return (
          <div key={i} className={`w-full group relative ${hlClass}`}>
            <span className={`text-xs ${dark ? 'text-slate-500' : 'text-slate-400'} mr-3 inline-block w-8 text-right`}>{lineNumber}</span>
            <span className="whitespace-pre">{ln}</span>

            {lineTooltips[lineNumber] && (
              <div className="absolute left-full ml-3 top-0 hidden group-hover:block max-w-xs z-20">
                <div className="bg-black/90 text-white text-xs p-2 rounded shadow-lg">{lineTooltips[lineNumber]}</div>
              </div>
            )}
          </div>
        );
      })}
    </pre>
  );
};

export default CodeHighlighter;
