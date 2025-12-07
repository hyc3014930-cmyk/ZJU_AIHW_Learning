import React from 'react';
import { LucideIcon } from 'lucide-react';

interface ConceptCardProps {
  title: string;
  icon: LucideIcon;
  description: string;
  points: string[];
  colorClass: string;
}

const ConceptCard: React.FC<ConceptCardProps> = ({ title, icon: Icon, description, points, colorClass }) => {
  return (
    <div className={`bg-white rounded-xl shadow-md border overflow-hidden hover:shadow-lg transition-shadow duration-300 ${colorClass}`}>
      <div className="p-5">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-white/80 shadow-sm">
             <Icon className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-slate-800">{title}</h3>
        </div>
        <p className="text-sm text-slate-600 mb-4 leading-relaxed">
          {description}
        </p>
        <ul className="space-y-2">
          {points.map((point, idx) => (
            <li key={idx} className="flex items-start gap-2 text-xs text-slate-700">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400 mt-1.5 flex-shrink-0" />
              <span>{point}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default ConceptCard;