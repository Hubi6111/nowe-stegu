"use client";

import { useRef, useState, useCallback } from "react";

interface BeforeAfterSliderProps {
  before: string;
  after: string;
  initialPosition?: number;
}

export default function BeforeAfterSlider({
  before,
  after,
  initialPosition = 50,
}: BeforeAfterSliderProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(initialPosition);
  const dragging = useRef(false);

  const updatePos = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.max(1, Math.min(99, pct)));
  }, []);

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      dragging.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      updatePos(e.clientX);
    },
    [updatePos]
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (dragging.current) updatePos(e.clientX);
    },
    [updatePos]
  );

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative select-none overflow-hidden rounded-xl border border-stone-200 touch-none"
      style={{ maxWidth: 900, touchAction: "none", WebkitUserSelect: "none" } as React.CSSProperties}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <img
        src={after}
        alt="Po wizualizacji"
        className="w-full block"
        draggable={false}
      />

      <img
        src={before}
        alt="Przed wizualizacją"
        className="absolute inset-0 w-full h-full object-cover block"
        draggable={false}
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      />

      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white z-10 pointer-events-none shadow-sm"
        style={{ left: `${pos}%` }}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-10 h-10 sm:w-9 sm:h-9 rounded-full bg-white shadow-lg flex items-center justify-center pointer-events-auto cursor-col-resize border border-stone-200">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path
              d="M5 3L1 8L5 13M11 3L15 8L11 13"
              stroke="#A01B1B"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </div>

      <span className="absolute top-2.5 left-2.5 px-2.5 py-1 text-[10px] font-semibold text-white bg-black/50 rounded-lg pointer-events-none backdrop-blur-sm">
        Przed
      </span>
      <span className="absolute top-2.5 right-2.5 px-2.5 py-1 text-[10px] font-semibold text-white bg-black/50 rounded-lg pointer-events-none backdrop-blur-sm">
        Po
      </span>
    </div>
  );
}
