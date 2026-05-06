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

/* Brick-like grid: alternating rows with offset */
const ROWS = [
  [0, 1, 2, 3],       // 4 bricks
  [4, 5, 6, 7, 8],    // 5 half-offset bricks
  [9, 10, 11, 12],    // 4 bricks
  [13, 14, 15, 16, 17], // 5 half-offset bricks
  [18, 19, 20, 21],   // 4 bricks
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
      {/* Brick grid animation */}
      <div className="flex flex-col gap-[3px]">
        {ROWS.map((row, rowIdx) => (
          <div
            key={rowIdx}
            className="flex gap-[3px]"
            style={{ paddingLeft: row.length === 5 ? 0 : 10 }}
          >
            {row.map((cellIdx) => (
              <div
                key={cellIdx}
                className="rounded-[2px] grid-loading-cell"
                style={{
                  width: row.length === 5 ? 16 : 21,
                  height: 10,
                  animationDelay: `${cellIdx * 0.1}s`,
                }}
              />
            ))}
          </div>
        ))}
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
    </div>
  );
}
