import React from 'react';
import { Image, FileText, Mic, Video, Code, MessageSquare, Sparkles } from 'lucide-react';

// Foundation Model - Multimodal capabilities
export const FoundationModelIllustration = () => {
  const traditionalModels = [
    { modality: 'Text', icon: FileText, models: ['BERT', 'GPT-2', 'T5'], color: 'brand' },
    { modality: 'Image', icon: Image, models: ['ResNet', 'VGG', 'YOLO'], color: 'stone-dark' },
    { modality: 'Audio', icon: Mic, models: ['Wav2Vec', 'DeepSpeech'], color: 'stone-med' },
    { modality: 'Video', icon: Video, models: ['I3D', 'SlowFast'], color: 'stone-light' },
    { modality: 'Code', icon: Code, models: ['CodeBERT', 'GraphCodeBERT'], color: 'brand-light' },
  ];

  const foundationCapabilities = [
    { input: 'Text', output: 'Text', example: '"Summarize this article..."' },
    { input: 'Image', output: 'Text', example: '"What\'s in this photo?"' },
    { input: 'Text', output: 'Image', example: '"Generate a sunset over mountains"' },
    { input: 'Audio', output: 'Text', example: '"Transcribe this meeting..."' },
    { input: 'Text', output: 'Code', example: '"Write a React component..."' },
    { input: 'Video', output: 'Text', example: '"Describe what happens..."' },
  ];

  return (
    <div className="my-10 space-y-8">
      {/* Traditional: Siloed Models */}
      <div className="bg-stone-100 rounded-2xl p-6 border border-stone-200">
        <div className="text-xs font-mono text-stone-500 uppercase tracking-wider mb-4">
          Traditional ML: One Model Per Task Per Modality
        </div>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {traditionalModels.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.modality} className="bg-white rounded-xl p-4 border border-stone-200 text-center">
                <div className={`w-10 h-10 mx-auto mb-2 rounded-full flex items-center justify-center
                  ${item.color === 'brand' ? 'bg-brand-100 text-brand-600' : ''}
                  ${item.color === 'stone-dark' ? 'bg-stone-200 text-stone-600' : ''}
                  ${item.color === 'stone-med' ? 'bg-stone-150 text-stone-500' : ''}
                  ${item.color === 'stone-light' ? 'bg-stone-100 text-stone-500' : ''}
                  ${item.color === 'brand-light' ? 'bg-brand-50 text-brand-500' : ''}
                `}>
                  <Icon size={20} />
                </div>
                <p className="font-semibold text-stone-800 text-sm mb-1">{item.modality}</p>
                <div className="space-y-1">
                  {item.models.map(m => (
                    <div key={m} className="text-xs font-mono bg-stone-100 px-2 py-0.5 rounded text-stone-500">
                      {m}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
        <div className="mt-4 p-3 bg-brand-50 rounded-xl border border-brand-200">
          <p className="text-sm text-brand-700">
            <strong>Problem:</strong> 5 modalities × multiple tasks = dozens of specialized models to train, deploy, and maintain
          </p>
        </div>
      </div>
      
      {/* Foundation Model: Unified */}
      <div className="bg-stone-900 rounded-2xl p-6 text-white overflow-hidden relative">
        <div className="text-xs font-mono text-stone-500 uppercase tracking-wider mb-4">
          Foundation Models: One Model, All Modalities
        </div>
        
        {/* Central model */}
        <div className="relative">
          {/* Input types - left side */}
          <div className="grid grid-cols-1 gap-2 md:absolute md:left-0 md:top-1/2 md:-translate-y-1/2 md:w-32">
            {['Text', 'Image', 'Audio', 'Video', 'Code'].map((type, i) => (
              <div key={type} className="flex items-center gap-2 px-3 py-1.5 bg-stone-800 rounded-lg border border-stone-700">
                <div className="w-2 h-2 rounded-full bg-brand-400" />
                <span className="text-xs text-stone-400">{type}</span>
              </div>
            ))}
          </div>
          
          {/* Center - The model */}
          <div className="my-6 md:my-0 md:mx-auto md:w-64 md:py-16">
            <div className="p-6 bg-gradient-to-br from-brand-500 to-brand-600 rounded-2xl text-center shadow-lg shadow-brand-500/20 relative">
              <Sparkles className="absolute top-3 right-3 w-5 h-5 text-brand-200" />
              <div className="text-2xl mb-2">🧠</div>
              <p className="font-bold text-lg">Foundation Model</p>
              <p className="text-brand-100 text-xs mt-1">Truly Multimodal</p>
            </div>
            <div className="flex justify-center gap-1 mt-3">
              <div className="text-stone-600">←</div>
              <div className="text-stone-500 text-xs">any input</div>
              <div className="text-stone-600">→</div>
              <div className="text-stone-500 text-xs">any output</div>
              <div className="text-stone-600">→</div>
            </div>
          </div>
          
          {/* Output types - right side */}
          <div className="grid grid-cols-1 gap-2 md:absolute md:right-0 md:top-1/2 md:-translate-y-1/2 md:w-32">
            {['Text', 'Image', 'Audio', 'Code', 'Structured'].map((type, i) => (
              <div key={type} className="flex items-center gap-2 px-3 py-1.5 bg-stone-800 rounded-lg border border-stone-700">
                <span className="text-xs text-stone-400">{type}</span>
                <div className="w-2 h-2 rounded-full bg-emerald-400 ml-auto" />
              </div>
            ))}
          </div>
        </div>
        
        {/* Example use cases */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-3 gap-2">
          {foundationCapabilities.map((cap, i) => (
            <div key={i} className="p-3 bg-stone-800/50 rounded-xl border border-stone-700">
              <div className="flex items-center gap-1 mb-1">
                <span className="text-xs px-1.5 py-0.5 bg-brand-500/20 text-brand-300 rounded">{cap.input}</span>
                <span className="text-stone-600">→</span>
                <span className="text-xs px-1.5 py-0.5 bg-emerald-500/20 text-emerald-300 rounded">{cap.output}</span>
              </div>
              <p className="text-xs text-stone-400 font-mono truncate">{cap.example}</p>
            </div>
          ))}
        </div>
        
        <div className="mt-6 p-3 bg-emerald-500/20 rounded-xl border border-emerald-500/40">
          <p className="text-sm text-emerald-300">
            <strong>Result:</strong> Any modality in → Any modality out. One API, infinite possibilities.
          </p>
        </div>
      </div>
    </div>
  );
};

// Metrics Dashboard
export const MetricsDashboardIllustration = () => (
  <div className="my-10 rounded-2xl overflow-hidden bg-gradient-to-br from-stone-900 via-stone-800 to-stone-900 p-8">
    <svg viewBox="0 0 800 350" className="w-full h-auto">
      <defs>
        <linearGradient id="barGreen" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#059669" />
        </linearGradient>
        <linearGradient id="barOrange" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#f97316" />
          <stop offset="100%" stopColor="#ea580c" />
        </linearGradient>
        <linearGradient id="barBlue" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#3b82f6" />
          <stop offset="100%" stopColor="#2563eb" />
        </linearGradient>
      </defs>
      
      {/* Dashboard frame */}
      <rect x="50" y="30" width="700" height="290" rx="12" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.1)" />
      
      {/* Header */}
      <rect x="50" y="30" width="700" height="40" rx="12" fill="rgba(255,255,255,0.05)" />
      <circle cx="75" cy="50" r="6" fill="#ef4444" opacity="0.8" />
      <circle cx="95" cy="50" r="6" fill="#f59e0b" opacity="0.8" />
      <circle cx="115" cy="50" r="6" fill="#10b981" opacity="0.8" />
      <text x="400" y="55" textAnchor="middle" fill="rgba(255,255,255,0.6)" fontSize="12" fontFamily="monospace">AI Performance Dashboard</text>
      
      {/* Metric cards */}
      {/* Card 1 - Accuracy */}
      <g transform="translate(80, 90)">
        <rect width="180" height="100" rx="8" fill="rgba(16,185,129,0.1)" stroke="rgba(16,185,129,0.3)" />
        <text x="15" y="30" fill="rgba(255,255,255,0.5)" fontSize="10" fontFamily="monospace">ACCURACY</text>
        <text x="15" y="60" fill="#10b981" fontSize="28" fontWeight="bold" fontFamily="system-ui">87.3%</text>
        <rect x="15" y="75" width="150" height="6" rx="3" fill="rgba(255,255,255,0.1)" />
        <rect x="15" y="75" width="130" height="6" rx="3" fill="url(#barGreen)" />
      </g>
      
      {/* Card 2 - Latency */}
      <g transform="translate(280, 90)">
        <rect width="180" height="100" rx="8" fill="rgba(249,115,22,0.1)" stroke="rgba(249,115,22,0.3)" />
        <text x="15" y="30" fill="rgba(255,255,255,0.5)" fontSize="10" fontFamily="monospace">LATENCY</text>
        <text x="15" y="60" fill="#f97316" fontSize="28" fontWeight="bold" fontFamily="system-ui">142ms</text>
        <rect x="15" y="75" width="150" height="6" rx="3" fill="rgba(255,255,255,0.1)" />
        <rect x="15" y="75" width="105" height="6" rx="3" fill="url(#barOrange)" />
      </g>
      
      {/* Card 3 - Throughput */}
      <g transform="translate(480, 90)">
        <rect width="180" height="100" rx="8" fill="rgba(59,130,246,0.1)" stroke="rgba(59,130,246,0.3)" />
        <text x="15" y="30" fill="rgba(255,255,255,0.5)" fontSize="10" fontFamily="monospace">THROUGHPUT</text>
        <text x="15" y="60" fill="#3b82f6" fontSize="28" fontWeight="bold" fontFamily="system-ui">2.4k/s</text>
        <rect x="15" y="75" width="150" height="6" rx="3" fill="rgba(255,255,255,0.1)" />
        <rect x="15" y="75" width="120" height="6" rx="3" fill="url(#barBlue)" />
      </g>
      
      {/* Chart area */}
      <g transform="translate(80, 210)">
        <text x="0" y="0" fill="rgba(255,255,255,0.4)" fontSize="10" fontFamily="monospace">SUCCESS RATE OVER TIME</text>
        
        {/* Mini chart */}
        <polyline 
          points="0,80 50,70 100,75 150,60 200,55 250,45 300,50 350,40 400,35 450,30 500,25 550,20"
          fill="none" 
          stroke="#10b981" 
          strokeWidth="2"
        />
        <polyline 
          points="0,80 50,70 100,75 150,60 200,55 250,45 300,50 350,40 400,35 450,30 500,25 550,20 550,90 0,90"
          fill="url(#barGreen)" 
          opacity="0.1"
        />
        
        {/* Axis */}
        <line x1="0" y1="90" x2="580" y2="90" stroke="rgba(255,255,255,0.1)" />
      </g>
    </svg>
  </div>
);

// CLIP Training Process
export const CLIPTrainingIllustration = () => (
  <div className="my-10 rounded-2xl overflow-hidden bg-gradient-to-br from-stone-900 via-stone-800 to-stone-900 p-8 md:p-12">
    <svg viewBox="0 0 800 400" className="w-full h-auto">
      <defs>
        <linearGradient id="matchGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#6ee7b7" />
        </linearGradient>
        <linearGradient id="mismatchGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#ef4444" />
          <stop offset="100%" stopColor="#fca5a5" />
        </linearGradient>
      </defs>
      
      {/* Title */}
      <text x="400" y="40" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="16" fontWeight="bold" fontFamily="system-ui">
        CLIP: Contrastive Language-Image Pre-training
      </text>
      
      {/* Image encoder box */}
      <g transform="translate(100, 100)">
        <rect width="200" height="200" rx="12" fill="rgba(59,130,246,0.1)" stroke="rgba(59,130,246,0.4)" strokeWidth="2" />
        <text x="100" y="30" textAnchor="middle" fill="#93c5fd" fontSize="12" fontFamily="monospace">IMAGE ENCODER</text>
        
        {/* Dog image placeholder */}
        <rect x="40" y="50" width="120" height="90" rx="8" fill="rgba(255,255,255,0.1)" />
        <text x="100" y="100" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="24">🐕</text>
        
        {/* Vector representation */}
        <rect x="40" y="155" width="120" height="30" rx="4" fill="rgba(59,130,246,0.3)" />
        <text x="100" y="175" textAnchor="middle" fill="#93c5fd" fontSize="10" fontFamily="monospace">[0.82, 0.15, ...]</text>
      </g>
      
      {/* Text encoder box */}
      <g transform="translate(500, 100)">
        <rect width="200" height="200" rx="12" fill="rgba(249,115,22,0.1)" stroke="rgba(249,115,22,0.4)" strokeWidth="2" />
        <text x="100" y="30" textAnchor="middle" fill="#fdba74" fontSize="12" fontFamily="monospace">TEXT ENCODER</text>
        
        {/* Text placeholder */}
        <rect x="20" y="60" width="160" height="70" rx="8" fill="rgba(255,255,255,0.1)" />
        <text x="100" y="90" textAnchor="middle" fill="rgba(255,255,255,0.6)" fontSize="11" fontFamily="system-ui">"a photo of</text>
        <text x="100" y="110" textAnchor="middle" fill="rgba(255,255,255,0.6)" fontSize="11" fontFamily="system-ui">a dog"</text>
        
        {/* Vector representation */}
        <rect x="40" y="155" width="120" height="30" rx="4" fill="rgba(249,115,22,0.3)" />
        <text x="100" y="175" textAnchor="middle" fill="#fdba74" fontSize="10" fontFamily="monospace">[0.79, 0.18, ...]</text>
      </g>
      
      {/* Matching connection (green) */}
      <g>
        <path d="M 300 200 Q 400 150 500 200" fill="none" stroke="url(#matchGrad)" strokeWidth="3" strokeDasharray="8,4">
          <animate attributeName="stroke-dashoffset" from="24" to="0" dur="1s" repeatCount="indefinite"/>
        </path>
        <circle cx="400" cy="160" r="20" fill="rgba(16,185,129,0.2)" stroke="#10b981" strokeWidth="2" />
        <text x="400" y="165" textAnchor="middle" fill="#10b981" fontSize="16">✓</text>
        <text x="400" y="130" textAnchor="middle" fill="#6ee7b7" fontSize="10" fontFamily="monospace">MATCH</text>
      </g>
      
      {/* Mismatched pair (bottom) */}
      <g transform="translate(0, 80)">
        <rect x="150" y="250" width="80" height="60" rx="6" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.2)" />
        <text x="190" y="285" textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="20">🐱</text>
        
        <rect x="570" y="250" width="80" height="60" rx="6" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.2)" />
        <text x="610" y="275" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9" fontFamily="system-ui">"a photo</text>
        <text x="610" y="290" textAnchor="middle" fill="rgba(255,255,255,0.4)" fontSize="9" fontFamily="system-ui">of a dog"</text>
        
        <path d="M 230 280 Q 400 320 570 280" fill="none" stroke="url(#mismatchGrad)" strokeWidth="2" strokeDasharray="4,4" opacity="0.5" />
        <circle cx="400" cy="305" r="15" fill="rgba(239,68,68,0.2)" stroke="#ef4444" strokeWidth="1.5" />
        <text x="400" y="310" textAnchor="middle" fill="#ef4444" fontSize="12">✗</text>
      </g>
      
      {/* Legend */}
      <g transform="translate(300, 380)">
        <circle cx="0" cy="0" r="6" fill="#10b981" />
        <text x="15" y="4" fill="rgba(255,255,255,0.5)" fontSize="10" fontFamily="system-ui">Pull matching pairs together</text>
        <circle cx="200" cy="0" r="6" fill="#ef4444" />
        <text x="215" y="4" fill="rgba(255,255,255,0.5)" fontSize="10" fontFamily="system-ui">Push mismatches apart</text>
      </g>
    </svg>
  </div>
);

// MCP Architecture Illustration
export const MCPArchitectureIllustration = () => (
  <div className="my-8">
    <svg viewBox="0 0 800 400" className="w-full h-auto max-w-3xl mx-auto">
      <defs>
        <linearGradient id="mcpClientGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#8b5cf6" />
          <stop offset="100%" stopColor="#6366f1" />
        </linearGradient>
        <linearGradient id="mcpServerGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#f97316" />
          <stop offset="100%" stopColor="#ea580c" />
        </linearGradient>
        <linearGradient id="mcpDataGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#059669" />
        </linearGradient>
        <filter id="mcpShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.15"/>
        </filter>
      </defs>
      
      {/* Background */}
      <rect width="800" height="400" fill="#fafaf9" rx="16" />
      
      {/* Title */}
      <text x="400" y="40" textAnchor="middle" fill="#44403c" fontSize="18" fontWeight="bold" fontFamily="system-ui">MCP Architecture</text>
      
      {/* MCP Client Box */}
      <g filter="url(#mcpShadow)">
        <rect x="50" y="120" width="180" height="160" rx="16" fill="url(#mcpClientGrad)" />
        <text x="140" y="165" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold" fontFamily="system-ui">MCP Client</text>
        <text x="140" y="190" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="12" fontFamily="system-ui">Claude Desktop</text>
        <text x="140" y="210" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="12" fontFamily="system-ui">Cursor</text>
        <text x="140" y="230" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="12" fontFamily="system-ui">Any MCP Client</text>
        <rect x="70" y="245" width="140" height="24" rx="6" fill="rgba(255,255,255,0.2)" />
        <text x="140" y="262" textAnchor="middle" fill="white" fontSize="11" fontFamily="system-ui">Sends requests</text>
      </g>
      
      {/* MCP Server Box */}
      <g filter="url(#mcpShadow)">
        <rect x="310" y="120" width="180" height="160" rx="16" fill="url(#mcpServerGrad)" />
        <text x="400" y="165" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold" fontFamily="system-ui">MCP Server</text>
        <text x="400" y="190" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="12" fontFamily="system-ui">Your Code</text>
        <rect x="330" y="205" width="140" height="24" rx="6" fill="rgba(255,255,255,0.2)" />
        <text x="400" y="222" textAnchor="middle" fill="white" fontSize="10" fontFamily="system-ui">📖 Resources (read)</text>
        <rect x="330" y="235" width="140" height="24" rx="6" fill="rgba(255,255,255,0.2)" />
        <text x="400" y="252" textAnchor="middle" fill="white" fontSize="10" fontFamily="system-ui">🔧 Tools (actions)</text>
      </g>
      
      {/* Data Source Box */}
      <g filter="url(#mcpShadow)">
        <rect x="570" y="120" width="180" height="160" rx="16" fill="url(#mcpDataGrad)" />
        <text x="660" y="165" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold" fontFamily="system-ui">Data Sources</text>
        <text x="660" y="195" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="12" fontFamily="system-ui">🗄️ Databases</text>
        <text x="660" y="220" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="12" fontFamily="system-ui">🌐 APIs</text>
        <text x="660" y="245" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="12" fontFamily="system-ui">📁 Files</text>
        <text x="660" y="270" textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize="12" fontFamily="system-ui">🔌 Services</text>
      </g>
      
      {/* Arrows */}
      <g>
        {/* Client to Server */}
        <path d="M 230 180 L 300 180" fill="none" stroke="#78716c" strokeWidth="3" markerEnd="url(#arrowhead)" />
        <path d="M 300 220 L 230 220" fill="none" stroke="#78716c" strokeWidth="3" markerEnd="url(#arrowhead)" />
        
        {/* Server to Data */}
        <path d="M 490 180 L 560 180" fill="none" stroke="#78716c" strokeWidth="3" markerEnd="url(#arrowhead)" />
        <path d="M 560 220 L 490 220" fill="none" stroke="#78716c" strokeWidth="3" markerEnd="url(#arrowhead)" />
        
        {/* Arrowhead marker */}
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#78716c" />
          </marker>
        </defs>
      </g>
      
      {/* Protocol Labels */}
      <rect x="240" y="145" width="80" height="20" rx="4" fill="#e7e5e4" />
      <text x="280" y="159" textAnchor="middle" fill="#57534e" fontSize="9" fontFamily="monospace">JSON-RPC</text>
      
      <rect x="500" y="145" width="50" height="20" rx="4" fill="#e7e5e4" />
      <text x="525" y="159" textAnchor="middle" fill="#57534e" fontSize="9" fontFamily="monospace">Any</text>
      
      {/* Transport info */}
      <g transform="translate(400, 330)">
        <rect x="-150" y="0" width="300" height="50" rx="8" fill="#f5f5f4" stroke="#d6d3d1" />
        <text x="0" y="20" textAnchor="middle" fill="#57534e" fontSize="11" fontFamily="system-ui" fontWeight="bold">Transport Options</text>
        <text x="-60" y="38" textAnchor="middle" fill="#78716c" fontSize="10" fontFamily="system-ui">stdio (local)</text>
        <text x="60" y="38" textAnchor="middle" fill="#78716c" fontSize="10" fontFamily="system-ui">SSE (remote)</text>
      </g>
    </svg>
  </div>
);

// Map prompts to illustrations
export const illustrationMap: Record<string, React.FC> = {
  'foundation_model': FoundationModelIllustration,
  'metrics_dashboard': MetricsDashboardIllustration,
  'clip_training': CLIPTrainingIllustration,
  'mcp_architecture': MCPArchitectureIllustration,
};

