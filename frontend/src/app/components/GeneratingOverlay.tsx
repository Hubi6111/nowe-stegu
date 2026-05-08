"use client";

import { useEffect, useState } from "react";

/* Two-stage pipeline messages */
const STAGE_1_MESSAGES = [
  "Analizuję zdjęcie pokoju…",
  "Rozpoznaję perspektywę ściany…",
  "Obliczam skalę i wymiary…",
  "Wykrywam oświetlenie sceny…",
];

const STAGE_2_MESSAGES = [
  "Nakładam teksturę na ścianę…",
  "Dopasowuję materiał do perspektywy…",
  "Dostosowuję oświetlenie i cienie…",
  "Zachowuję obiekty na pierwszym planie…",
  "Finalizuję wizualizację…",
];

/* Time estimate: ~12s for analysis, ~18s for render */
const STAGE_1_DURATION_S = 12;

export default function GeneratingOverlay() {
  const [msgIdx, setMsgIdx] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  const isStage2 = elapsed >= STAGE_1_DURATION_S;
  const currentMessages = isStage2 ? STAGE_2_MESSAGES : STAGE_1_MESSAGES;

  useEffect(() => {
    let localIdx = 0;
    const iv = setInterval(() => {
      localIdx++;
      setMsgIdx(localIdx);
    }, 3500);
    return () => clearInterval(iv);
  }, []);

  // Elapsed time counter
  useEffect(() => {
    const iv = setInterval(() => {
      setElapsed(prev => prev + 1);
    }, 1000);
    return () => clearInterval(iv);
  }, []);

  // Pick a message from the appropriate stage array
  const displayMsg = currentMessages[msgIdx % currentMessages.length];

  return (
    <div className="animate-fade-in flex flex-col items-center gap-6 py-10 sm:py-16">
      {/* Stage indicator */}
      <div className="flex items-center gap-2 sm:gap-3 text-[10px] sm:text-[11px] font-medium flex-wrap justify-center">
        <div className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full transition-all ${
          !isStage2
            ? "bg-[#A01B1B]/10 text-[#A01B1B] border border-[#A01B1B]/20"
            : "bg-stone-100 text-stone-400 border border-stone-200"
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${!isStage2 ? "bg-[#A01B1B] animate-pulse" : "bg-stone-300"}`} />
          <span className="whitespace-nowrap">Analiza sceny</span>
        </div>
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-stone-300 shrink-0">
          <path d="M4 2l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className={`flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-full transition-all ${
          isStage2
            ? "bg-[#A01B1B]/10 text-[#A01B1B] border border-[#A01B1B]/20"
            : "bg-stone-100 text-stone-400 border border-stone-200"
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${isStage2 ? "bg-[#A01B1B] animate-pulse" : "bg-stone-300"}`} />
          <span className="whitespace-nowrap">Render tekstury</span>
        </div>
      </div>

      {/* Animated loader */}
      <div className="relative w-20 h-20">
        {/* Outer ring */}
        <div className="absolute inset-0 rounded-full border-[3px] border-stone-200" />
        <div
          className="absolute inset-0 rounded-full border-[3px] border-transparent border-t-[#A01B1B] animate-loader-spin"
        />
        {/* Inner ring */}
        <div className="absolute inset-2.5 rounded-full border-[2px] border-stone-100" />
        <div
          className="absolute inset-2.5 rounded-full border-[2px] border-transparent border-b-[#A01B1B]/60 animate-loader-spin-reverse"
        />
        {/* Center — stage number */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs font-bold text-[#A01B1B]/60">{isStage2 ? "2" : "1"}</span>
        </div>
      </div>

      {/* Status text */}
      <div className="text-center min-h-[3rem] px-4">
        <p
          key={`${isStage2 ? "s2" : "s1"}-${msgIdx}`}
          className="text-sm font-medium text-stone-600 animate-status-text"
        >
          {displayMsg}
        </p>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-xs px-4">
        <div className="h-1.5 bg-stone-100 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-[#A01B1B] to-[#C84848] rounded-full animate-progress" />
        </div>
        <p className="text-[10px] text-stone-400 mt-1.5 text-center">
          Dwuetapowa wizualizacja AI — zwykle ~20-40 sekund
        </p>
      </div>

      {/* Grid animation */}
      <div className="grid grid-cols-5 gap-1 mt-2">
        {Array.from({ length: 15 }).map((_, i) => (
          <div
            key={i}
            className="w-3 h-3 rounded-sm grid-loading-cell"
            style={{ animationDelay: `${i * 0.12}s` }}
          />
        ))}
      </div>
    </div>
  );
}
