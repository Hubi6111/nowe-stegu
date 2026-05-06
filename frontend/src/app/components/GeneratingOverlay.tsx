"use client";

import { useEffect, useState } from "react";

const STATUS_MESSAGES = [
  "Analizuję zdjęcie pokoju…",
  "Dopasowuję perspektywę ściany…",
  "Skaluję teksturę…",
  "Nakładam materiał na ścianę…",
  "Dostosowuję oświetlenie…",
  "Zachowuję obiekty na pierwszym planie…",
  "Finalizuję wizualizację…",
];

export default function GeneratingOverlay() {
  const [msgIdx, setMsgIdx] = useState(0);

  useEffect(() => {
    const iv = setInterval(() => {
      setMsgIdx(prev => (prev + 1) % STATUS_MESSAGES.length);
    }, 3500);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="animate-fade-in flex flex-col items-center gap-6 py-10 sm:py-16">
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
        {/* Center dot */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-3 h-3 rounded-full bg-[#A01B1B]/20 animate-pulse-glow" />
        </div>
      </div>

      {/* Status text */}
      <div className="text-center min-h-[3rem]">
        <p
          key={msgIdx}
          className="text-sm font-medium text-stone-600 animate-status-text"
        >
          {STATUS_MESSAGES[msgIdx]}
        </p>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-xs">
        <div className="h-1.5 bg-stone-100 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-[#A01B1B] to-[#C84848] rounded-full animate-progress" />
        </div>
        <p className="text-[10px] text-stone-400 mt-1.5 text-center">
          Wizualizacja AI — zwykle ~15-30 sekund
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
