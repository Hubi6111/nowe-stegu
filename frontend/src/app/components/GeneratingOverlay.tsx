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

/* ── Brick wall layout ── */
const WALL_ROWS = [
  // Row 0 — full bricks
  [
    { w: 3, h: 1, x: 0, y: 0 },
    { w: 3, h: 1, x: 3.2, y: 0 },
    { w: 3, h: 1, x: 6.4, y: 0 },
    { w: 3, h: 1, x: 9.6, y: 0 },
  ],
  // Row 1 — offset
  [
    { w: 1.5, h: 1, x: 0, y: 1.2 },
    { w: 3,   h: 1, x: 1.7, y: 1.2 },
    { w: 3,   h: 1, x: 4.9, y: 1.2 },
    { w: 3,   h: 1, x: 8.1, y: 1.2 },
    { w: 1.5, h: 1, x: 11.3, y: 1.2 },
  ],
  // Row 2 — full bricks
  [
    { w: 3, h: 1, x: 0, y: 2.4 },
    { w: 3, h: 1, x: 3.2, y: 2.4 },
    { w: 3, h: 1, x: 6.4, y: 2.4 },
    { w: 3, h: 1, x: 9.6, y: 2.4 },
  ],
  // Row 3 — offset
  [
    { w: 1.5, h: 1, x: 0, y: 3.6 },
    { w: 3,   h: 1, x: 1.7, y: 3.6 },
    { w: 3,   h: 1, x: 4.9, y: 3.6 },
    { w: 3,   h: 1, x: 8.1, y: 3.6 },
    { w: 1.5, h: 1, x: 11.3, y: 3.6 },
  ],
  // Row 4 — full bricks
  [
    { w: 3, h: 1, x: 0, y: 4.8 },
    { w: 3, h: 1, x: 3.2, y: 4.8 },
    { w: 3, h: 1, x: 6.4, y: 4.8 },
    { w: 3, h: 1, x: 9.6, y: 4.8 },
  ],
];

const ALL_BRICKS = WALL_ROWS.flat();
const UNIT = 16; // px per unit

export default function GeneratingOverlay() {
  const [msgIdx, setMsgIdx] = useState(0);
  const [visibleCount, setVisibleCount] = useState(0);

  // Cycle status messages
  useEffect(() => {
    const iv = setInterval(() => {
      setMsgIdx(prev => (prev + 1) % STATUS_MESSAGES.length);
    }, 3500);
    return () => clearInterval(iv);
  }, []);

  // Animate bricks appearing bottom-to-top
  useEffect(() => {
    if (visibleCount >= ALL_BRICKS.length) return;
    const timeout = setTimeout(() => {
      setVisibleCount(prev => prev + 1);
    }, 180 + Math.random() * 120);
    return () => clearTimeout(timeout);
  }, [visibleCount]);

  // Reverse order so bottom row appears first
  const sortedBricks = [...ALL_BRICKS]
    .map((b, i) => ({ ...b, idx: i }))
    .sort((a, b) => b.y - a.y || a.x - b.x);

  const wallWidth = 12.8 * UNIT;
  const wallHeight = 5.8 * UNIT;

  return (
    <div className="animate-fade-in flex flex-col items-center gap-6 py-10 sm:py-16">
      {/* Brick wall animation */}
      <div
        className="relative"
        style={{ width: wallWidth, height: wallHeight }}
      >
        {sortedBricks.map((brick, sortIdx) => {
          const isVisible = sortIdx < visibleCount;
          return (
            <div
              key={brick.idx}
              className="absolute rounded-[3px] transition-all"
              style={{
                left: brick.x * UNIT,
                top: brick.y * UNIT,
                width: brick.w * UNIT,
                height: brick.h * UNIT,
                backgroundColor: isVisible ? `hsl(${8 + (brick.idx % 5) * 3}, ${55 + (brick.idx % 3) * 10}%, ${42 + (brick.idx % 7) * 3}%)` : "transparent",
                border: isVisible ? "1px solid hsl(30, 20%, 35%)" : "1px dashed hsl(30, 10%, 80%)",
                opacity: isVisible ? 1 : 0.3,
                transform: isVisible ? "translateY(0) scale(1)" : "translateY(8px) scale(0.9)",
                transition: `all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) ${sortIdx * 0.02}s`,
                boxShadow: isVisible ? "inset 0 1px 0 rgba(255,255,255,0.15), inset 0 -1px 0 rgba(0,0,0,0.15)" : "none",
              }}
            />
          );
        })}
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
