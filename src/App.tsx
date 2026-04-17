import React, { useState, useCallback, useRef } from 'react';
import { 
  Upload, 
  ShieldCheck, 
  AlertTriangle, 
  Info, 
  Scan, 
  RefreshCw, 
  FileText, 
  Activity,
  CheckCircle2,
  X,
  Fingerprint
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import Markdown from 'react-markdown';
import { cn } from './lib/utils';
import { analyzeMedia, AnalysisResult } from './services/gemini';

type AppState = 'IDLE' | 'ANALYZING' | 'RESULT';

export default function App() {
  const [state, setState] = useState<AppState>('IDLE');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startAnalysis = useCallback(async (base64: string, mimeType: string) => {
    setState('ANALYZING');
    setError(null);
    try {
      console.log('Initiating forensic analysis...', { mimeType, size: base64.length });
      const analysisResult = await analyzeMedia(base64, mimeType);
      setResult(analysisResult);
      setState('RESULT');
    } catch (err: any) {
      console.error('Forensic Engine Error:', err);
      setError(err.message || 'Verification failed. Please check your connection and try again.');
      setState('IDLE');
    }
  }, []);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/') && !file.type.startsWith('video/')) {
      setError('Unsupported format. Please upload an image or video.');
      return;
    }

    if (file.size > 20 * 1024 * 1024) { // Increase to 20MB for better video support
      setError('File too large. Max size is 20MB.');
      return;
    }

    setError(null);
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      setPreviewUrl(result);
      startAnalysis(result, file.type);
    };
    reader.onerror = () => {
      setError('Failed to read file from local storage.');
    };
    reader.readAsDataURL(file);
  }, [startAnalysis]);

  const reset = () => {
    setState('IDLE');
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-[#E4E4E7] font-sans selection:bg-blue-500/30">
      {/* Decorative Grid Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden opacity-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
        <div className="absolute inset-0 bg-radial-[circle_at_50%_50%] from-blue-500/10 via-transparent to-transparent"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <header className="flex items-center justify-between mb-12 border-b border-white/5 pb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded flex items-center justify-center shadow-[0_0_20px_rgba(37,99,235,0.4)]">
              <Fingerprint className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                DEEPSCAN <span className="text-blue-500">AI</span>
              </h1>
              <p className="text-[10px] uppercase tracking-[0.2em] font-mono text-white/40">Forensic Authentication Engine v3.0</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-6 text-[10px] font-mono uppercase tracking-wider text-white/40">
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              SYSTEM_READY
            </div>
            <div>Uptime: 99.9%</div>
            <div>Nodes: 12 Active</div>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Input/Preview */}
          <div className="lg:col-span-12 xl:col-span-7">
            <AnimatePresence mode="wait">
              {state === 'IDLE' ? (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.98 }}
                  className="relative"
                >
                  <div 
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={onDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className={cn(
                      "group relative aspect-video border-2 border-dashed border-white/10 rounded-2xl bg-white/[0.02] flex flex-col items-center justify-center cursor-pointer transition-all hover:bg-white/[0.04] hover:border-blue-500/50",
                      error && "border-red-500/50 bg-red-500/5"
                    )}
                  >
                    <input 
                      type="file" 
                      className="hidden" 
                      ref={fileInputRef} 
                      onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                      accept="image/*,video/*"
                    />
                    
                    <div className="flex flex-col items-center gap-4 text-center max-w-sm px-8">
                      <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center transition-transform group-hover:scale-110">
                        <Upload className="w-8 h-8 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-lg font-medium text-white">Upload Media for Analysis</h3>
                        <p className="text-sm text-white/40 mt-1">Drag and drop images or videos to check for digital manipulation, AI generation, and synthetic artifacts.</p>
                      </div>
                      <div className="mt-2 py-2 px-4 rounded-full bg-blue-500/10 border border-blue-500/20 text-[11px] font-mono text-blue-400 uppercase tracking-wider">
                        Select File to Start
                      </div>
                    </div>

                    {error && (
                      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 text-red-400 text-xs bg-red-500/10 px-4 py-2 rounded-lg border border-red-500/20">
                        <AlertTriangle className="w-4 h-4" />
                        {error}
                      </div>
                    )}
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="active"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="relative aspect-video rounded-2xl overflow-hidden bg-black border border-white/10 shadow-2xl group flex items-center justify-center"
                >
                  {previewUrl && (
                    selectedFile?.type.startsWith('video/') ? (
                      <video 
                        src={previewUrl} 
                        className="max-w-full max-h-full" 
                        controls 
                        autoPlay 
                        muted 
                        loop
                      />
                    ) : (
                      <img 
                        src={previewUrl} 
                        alt="Preview" 
                        className="w-full h-full object-contain"
                      />
                    )
                  )}

                  {/* Scanning Animation */}
                  {state === 'ANALYZING' && (
                    <div className="absolute inset-0 overflow-hidden pointer-events-none">
                      {/* Face Target Box (Decorative) */}
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: [0, 1, 0, 1, 0.5], scale: [0.8, 1.1, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity, repeatType: 'reverse' }}
                        className="absolute top-1/4 left-1/3 w-1/3 h-1/2 border-2 border-blue-400/40 rounded-3xl z-30"
                      >
                         <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-blue-400" />
                         <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-blue-400" />
                         <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-blue-400" />
                         <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-blue-400" />
                         <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-[10px] font-mono text-blue-400 bg-blue-500/10 px-2 py-1 rounded backdrop-blur-sm border border-blue-500/20">
                               LOCKING_ON_BIOMETRICS
                            </div>
                         </div>
                      </motion.div>

                      <motion.div 
                        initial={{ top: '-10%' }}
                        animate={{ top: '110%' }}
                        transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                        className="absolute w-full h-0.5 bg-blue-400 shadow-[0_0_25px_4px_rgba(59,130,246,0.8)] z-20"
                      />
                      <div className="absolute inset-0 bg-blue-500/5 mix-blend-overlay" />
                    </div>
                  )}

                  {/* Top Bar for Preview */}
                  <div className="absolute top-0 inset-x-0 p-4 bg-gradient-to-b from-black/80 to-transparent flex items-center justify-between opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="flex items-center gap-2">
                       <div className="text-[10px] font-mono text-white/50 bg-black/40 px-2 py-1 rounded border border-white/10">PREVIEW_IMG_MAPPED</div>
                    </div>
                    <button 
                      onClick={reset}
                      className="p-2 rounded-full bg-black/50 hover:bg-red-500/20 text-white/60 hover:text-red-400 transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Info Cards Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
              <div className="p-5 rounded-2xl border border-white/5 bg-white/[0.02]">
                <div className="flex items-center gap-3 mb-3">
                  <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
                    <Scan className="w-5 h-5" />
                  </div>
                  <h4 className="text-sm font-semibold uppercase tracking-wider">Multi-Modal Scan</h4>
                </div>
                <p className="text-xs text-sh-neutral-400 leading-relaxed text-white/40">
                  Our neural engine analyzes high-frequency noise, facial geometry inconsistencies, and adversarial perturbation patterns common in GAN and Diffusion models.
                </p>
              </div>
              <div className="p-5 rounded-2xl border border-white/5 bg-white/[0.02]">
                <div className="flex items-center gap-3 mb-3">
                  <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                    <Activity className="w-5 h-5" />
                  </div>
                  <h4 className="text-sm font-semibold uppercase tracking-wider">Real-time Feed</h4>
                </div>
                <p className="text-xs text-sh-neutral-400 leading-relaxed text-white/40">
                  Connect direct API feeds for stream monitoring. DeepScan detects video frame injection and localized pixel manipulation in under 500ms.
                </p>
              </div>
            </div>
          </div>

          {/* Right Column: Status/Results */}
          <div className="lg:col-span-12 xl:col-span-5 h-full">
            <div className="sticky top-8">
              <AnimatePresence mode="wait">
                {state === 'ANALYZING' && (
                  <motion.div
                    key="analyzing"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="p-8 rounded-3xl border border-blue-500/20 bg-blue-500/5 backdrop-blur-sm"
                  >
                    <div className="flex flex-col items-center text-center">
                      <div className="relative w-24 h-24 mb-6">
                        <motion.div 
                          className="absolute inset-0 rounded-full border-4 border-blue-500/20"
                        />
                        <motion.div 
                          initial={{ pathLength: 0 }}
                          animate={{ pathLength: 1, rotate: 360 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                          className="absolute inset-0 rounded-full border-4 border-t-blue-400 border-r-transparent border-b-transparent border-l-transparent"
                        />
                        <div className="absolute inset-0 flex items-center justify-center">
                          <Activity className="w-10 h-10 text-blue-400 animate-pulse" />
                        </div>
                      </div>
                      <h2 className="text-xl font-bold text-white mb-2">Analyzing Pixels...</h2>
                      <div className="w-full max-w-[200px] h-1 bg-white/5 rounded-full overflow-hidden mt-4">
                        <motion.div 
                           className="h-full bg-blue-500"
                           animate={{ width: ["0%", "45%", "60%", "95%"] }}
                           transition={{ duration: 15, times: [0, 0.2, 0.5, 1] }}
                        />
                      </div>
                      <div className="mt-8 space-y-4 w-full">
                        {[
                          'Isolating face clusters...',
                          'Mapping iris patterns...',
                          'Checking skin porosity coefficients...',
                          'Scanning hairline boundaries...',
                          'Validating temporal expression sync...'
                        ].map((step, i) => (
                          <div key={step} className="flex items-center gap-3 text-xs font-mono text-white/30">
                            <motion.div 
                              animate={{ opacity: [0.3, 1, 0.3] }}
                              transition={{ duration: 1.5, delay: i * 0.4, repeat: Infinity }}
                              className="w-1.5 h-1.5 rounded-full bg-blue-400"
                            />
                            {step}
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}

                {state === 'RESULT' && result && (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex flex-col gap-6"
                  >
                    {/* Score Card */}
                    <div className={cn(
                      "p-8 rounded-3xl border shadow-[0_20px_50px_rgba(0,0,0,0.5)]",
                      result.isReal ? "bg-green-500/5 border-green-500/20" : "bg-red-500/5 border-red-500/20"
                    )}>
                      <div className="flex items-center justify-between mb-8">
                        <div>
                          <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/40 mb-1">Inference Result</p>
                          <h2 className={cn(
                            "text-4xl font-black tracking-tighter",
                            result.isReal ? "text-green-400" : "text-red-400"
                          )}>
                            {result.isReal ? "REAL" : "FAKE"}
                          </h2>
                          <p className="text-[10px] font-mono uppercase text-white/30 mt-1">{result.verdict}</p>
                        </div>
                        <div className="text-right">
                           <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-white/40 mb-1">Confidence</p>
                           <p className="text-xl font-bold text-white">{result.confidence}%</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 mb-8">
                        <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                          <p className="text-[10px] font-mono uppercase text-white/40 mb-2">Deepfake Score</p>
                          <div className="flex items-end gap-2">
                             <span className="text-3xl font-bold text-white">{result.score}</span>
                             <span className="text-white/20 text-sm mb-1">/ 100</span>
                          </div>
                        </div>
                        <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                          <p className="text-[10px] font-mono uppercase text-white/40 mb-2">Analysis Status</p>
                          <div className="flex items-center gap-2 mt-2">
                            <CheckCircle2 className="w-5 h-5 text-blue-400" />
                            <span className="text-xs font-bold uppercase tracking-wider text-blue-400">Complete</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col gap-3">
                         <h4 className="text-[10px] font-mono uppercase tracking-widest text-white/40">Detected Anomalies</h4>
                         <div className="space-y-2">
                            {result.findings.map((finding, idx) => (
                              <div key={idx} className="flex items-start gap-3 p-3 rounded-xl bg-white/5 border border-white/5 text-xs text-white/70">
                                <Info className="w-4 h-4 text-blue-400 shrink-0" />
                                {finding}
                              </div>
                            ))}
                         </div>
                      </div>
                    </div>

                    {/* Report Content */}
                    <div className="p-8 rounded-3xl border border-white/5 bg-white/[0.02] overflow-hidden">
                       <div className="flex items-center gap-2 mb-6 text-white/60">
                         <FileText className="w-4 h-4" />
                         <span className="text-[10px] font-mono uppercase tracking-[0.2em]">Forensics Detail Report</span>
                       </div>
                       <div className="prose prose-invert prose-xs max-w-none text-white/50 leading-relaxed font-sans text-sm">
                          <Markdown>{result.explanation}</Markdown>
                       </div>
                       
                       <button 
                         onClick={reset}
                         className="w-full mt-8 py-4 px-6 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 text-white font-medium transition-all flex items-center justify-center gap-2 uppercase tracking-widest text-[10px]"
                       >
                         <RefreshCw className="w-3 h-3" />
                         Start New Analysis
                       </button>
                    </div>
                  </motion.div>
                )}

                {state === 'IDLE' && (
                  <motion.div
                    key="waiting"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="h-full flex flex-col justify-center gap-6 p-8 border border-white/5 bg-white/[0.01] rounded-3xl"
                  >
                    <div className="space-y-6">
                      <div className="flex items-start gap-4">
                        <div className="p-3 rounded-xl bg-blue-500/10 text-blue-400 shrink-0">
                          <ShieldCheck className="w-6 h-6" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-white mb-1">Authenticity Guard</h4>
                          <p className="text-sm text-white/40 leading-relaxed">Verification process utilizes billion-parameter vision models trained on state-of-the-art synthetic dataset.</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-4">
                        <div className="p-3 rounded-xl bg-blue-500/10 text-blue-400 shrink-0">
                          <FileText className="w-6 h-6" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-white mb-1">Integrity Reports</h4>
                          <p className="text-sm text-white/40 leading-relaxed">Download detailed checksum and forensic proofs for legal and verification documentation requirements.</p>
                        </div>
                      </div>
                    </div>

                    <div className="mt-8 pt-8 border-t border-white/5">
                      <div className="text-[10px] font-mono text-white/30 uppercase mb-4 tracking-widest">System Capabilities</div>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          'Noise Analysis', 'Geometry Mapping', 'Pixel Diffusion', 
                          'Artifact Hunt', 'Author ID', 'EXIF Scrub'
                        ].map(tag => (
                          <div key={tag} className="px-3 py-1.5 rounded-lg bg-white/5 text-[9px] font-mono text-white/40 border border-white/5 uppercase">
                            {tag}
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </main>
      </div>

      {/* Footer */}
      <footer className="mt-20 border-t border-white/5 py-12 text-center relative z-10">
        <p className="text-[10px] font-mono uppercase tracking-[0.3em] text-white/20">
          PROTECTING REALITY THROUGH INTELLIGENT VERIFICATION. © 2026 DEEPSCAN SYSTEMS
        </p>
      </footer>
    </div>
  );
}
