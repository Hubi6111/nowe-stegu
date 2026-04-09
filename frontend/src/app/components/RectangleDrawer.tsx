"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Stage, Layer, Image as KonvaImage, Rect, Circle } from "react-konva";
import type Konva from "konva";

interface Point { x: number; y: number }

interface RectangleDrawerProps {
  imageSrc: string;
  points: Point[];
  onPointsChange: (points: Point[]) => void;
  onStageSizeChange?: (size: { width: number; height: number }) => void;
}

const IS_TOUCH = typeof window !== "undefined" && "ontouchstart" in window;
const HANDLE_R = IS_TOUCH ? 11 : 8;
const STEGU_RED = "#A01B1B";
const STEGU_RED_DARK = "#7A1515";
const EDGE_COLOR = "rgba(160, 27, 27, 0.75)";
const FILL_COLOR = "rgba(160, 27, 27, 0.10)";
const ZOOM_SIZE = 130;
const ZOOM_SCALE = 3;

export default function RectangleDrawer({
  imageSrc,
  points,
  onPointsChange,
  onStageSizeChange,
}: RectangleDrawerProps) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [stageSize, setStageSize] = useState({ width: 600, height: 400 });
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // Drawing state: mousedown sets origin, mousemove sets current → rectangle
  const [drawing, setDrawing] = useState(false);
  const [drawOrigin, setDrawOrigin] = useState<Point | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<Point | null>(null);

  // Corner dragging
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const [cursorPos, setCursorPos] = useState<Point | null>(null);

  const [hoverPos, setHoverPos] = useState<Point | null>(null);
  const isInteracting = drawing || draggingIdx !== null;
  const hasNoRect = points.length < 2;

  const fitImage = useCallback((img: HTMLImageElement) => {
    const containerW = containerRef.current?.clientWidth || window.innerWidth - 24;
    const maxW = Math.min(containerW, 1200);
    // Always scale to fit width; height follows proportionally
    const s = maxW / img.width;
    const newSize = { width: Math.round(img.width * s), height: Math.round(img.height * s) };
    setStageSize(newSize);
    onStageSizeChange?.(newSize);
    return newSize;
  }, [onStageSizeChange]);

  useEffect(() => {
    setImage(null);
    setLoadError(false);
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      fitImage(img);
      setImage(img);
      // Refit after layout settles (container may not have correct width on first call)
      requestAnimationFrame(() => { fitImage(img); });
    };
    img.onerror = () => setLoadError(true);
    img.src = imageSrc;
  }, [imageSrc, fitImage]);

  useEffect(() => {
    const h = () => { if (imgRef.current) fitImage(imgRef.current); };
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, [fitImage]);

  // ── Rect helpers ──
  const clamp = useCallback((p: Point) => ({
    x: Math.max(0, Math.min(stageSize.width, p.x)),
    y: Math.max(0, Math.min(stageSize.height, p.y)),
  }), [stageSize]);

  const rectFromPts = useCallback((a: Point, b: Point) => ({
    x: Math.min(a.x, b.x), y: Math.min(a.y, b.y),
    w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y),
  }), []);

  const cornersOf = useCallback((a: Point, b: Point): Point[] => {
    const r = rectFromPts(a, b);
    return [
      { x: r.x, y: r.y }, { x: r.x + r.w, y: r.y },
      { x: r.x + r.w, y: r.y + r.h }, { x: r.x, y: r.y + r.h },
    ];
  }, [rectFromPts]);

  // ── Pointer helpers (stage-relative) ──
  const getStagePos = useCallback((e: Konva.KonvaEventObject<MouseEvent | TouchEvent>): Point | null => {
    const stage = e.target.getStage();
    return stage?.getPointerPosition() ?? null;
  }, []);

  // ── Draw: mousedown → drag → mouseup ──
  const handleMouseDown = useCallback((e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (e.evt?.cancelable) e.evt.preventDefault();
    const isOnCanvas = e.target === e.target.getStage() || e.target.getClassName() === "Image";
    if (!isOnCanvas) return;
    const pos = getStagePos(e);
    if (!pos) return;
    setDrawing(true);
    setDrawOrigin(pos);
    setDrawCurrent(pos);
    setCursorPos(pos);
  }, [getStagePos]);

  const handleMouseMove = useCallback((e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (e.evt?.cancelable) e.evt.preventDefault();
    const pos = getStagePos(e);
    if (!pos) return;
    if (drawing && drawOrigin) {
      const clamped = clamp(pos);
      setDrawCurrent(clamped);
      setCursorPos(clamped);
    } else if (!drawing && !draggingIdx) {
      setHoverPos(pos);
    }
  }, [drawing, drawOrigin, draggingIdx, getStagePos, clamp]);

  const handleMouseUp = useCallback((e: Konva.KonvaEventObject<MouseEvent | TouchEvent>) => {
    if (e.evt?.cancelable) e.evt.preventDefault();
    if (drawing && drawOrigin && drawCurrent) {
      const r = rectFromPts(drawOrigin, drawCurrent);
      if (r.w > 5 && r.h > 5) {
        onPointsChange([
          { x: Math.min(drawOrigin.x, drawCurrent.x), y: Math.min(drawOrigin.y, drawCurrent.y) },
          { x: Math.max(drawOrigin.x, drawCurrent.x), y: Math.max(drawOrigin.y, drawCurrent.y) },
        ]);
      }
    }
    setDrawing(false);
    setDrawOrigin(null);
    setDrawCurrent(null);
    setCursorPos(null);
  }, [drawing, drawOrigin, drawCurrent, rectFromPts, onPointsChange]);

  // ── Corner drag (adjust existing rect) ──
  const handleCornerDrag = useCallback((idx: number, nx: number, ny: number) => {
    if (points.length < 2) return;
    const cx = Math.max(0, Math.min(stageSize.width, nx));
    const cy = Math.max(0, Math.min(stageSize.height, ny));
    let p0 = { ...points[0] }, p1 = { ...points[1] };

    if (idx === 0) { p0 = { x: cx, y: cy }; }
    else if (idx === 1) { p0 = { ...p0, y: cy }; p1 = { ...p1, x: cx }; }
    else if (idx === 2) { p1 = { x: cx, y: cy }; }
    else { p0 = { ...p0, x: cx }; p1 = { ...p1, y: cy }; }

    const nA = { x: Math.min(p0.x, p1.x), y: Math.min(p0.y, p1.y) };
    const nB = { x: Math.max(p0.x, p1.x), y: Math.max(p0.y, p1.y) };
    onPointsChange([nA, nB]);
    setCursorPos({ x: cx, y: cy });
  }, [points, onPointsChange, stageSize]);

  // ── Computed display rect ──
  const hasRect = points.length >= 2;
  const liveOrigin = drawing && drawOrigin ? drawOrigin : hasRect ? points[0] : null;
  const liveEnd = drawing && drawCurrent ? drawCurrent : hasRect ? points[1] : null;
  const displayRect = liveOrigin && liveEnd ? rectFromPts(liveOrigin, liveEnd) : null;
  const displayCorners = liveOrigin && liveEnd ? cornersOf(liveOrigin, liveEnd) : [];

  if (loadError) {
    return (
      <div className="flex items-center justify-center bg-red-50 rounded-2xl border border-red-200 h-48">
        <p className="text-red-600 text-sm">Nie udało się wczytać obrazu</p>
      </div>
    );
  }
  if (!image) {
    return (
      <div className="flex items-center justify-center rounded-2xl h-64 animate-shimmer">
        <p className="text-stone-400 text-sm">Ładowanie…</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative select-none w-full flex justify-center">
      <Stage
        ref={stageRef}
        width={stageSize.width}
        height={stageSize.height}
        onMouseDown={handleMouseDown}
        onTouchStart={handleMouseDown}
        onMouseMove={handleMouseMove}
        onTouchMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onTouchEnd={handleMouseUp}
        onMouseLeave={() => setHoverPos(null)}
        className="rounded-2xl overflow-hidden border border-stone-200 cursor-crosshair touch-none"
        style={{ background: "#F5F5F0", touchAction: "none", WebkitUserSelect: "none", userSelect: "none", WebkitTouchCallout: "none" } as React.CSSProperties}
      >
        <Layer>
          <KonvaImage image={image} width={stageSize.width} height={stageSize.height} />

          {displayRect && displayRect.w > 1 && displayRect.h > 1 && (
            <Rect
              x={displayRect.x} y={displayRect.y}
              width={displayRect.w} height={displayRect.h}
              fill={FILL_COLOR} stroke={EDGE_COLOR} strokeWidth={2}
              dash={drawing ? [6, 3] : []}
              cornerRadius={1}
            />
          )}

          {!drawing && displayCorners.map((pt, i) => (
            <Circle
              key={i}
              x={pt.x} y={pt.y}
              radius={HANDLE_R}
              fill={draggingIdx === i ? STEGU_RED_DARK : STEGU_RED}
              stroke="white" strokeWidth={2.5}
              shadowColor="rgba(0,0,0,0.3)" shadowBlur={5} shadowOffsetY={2}
              draggable
              hitStrokeWidth={IS_TOUCH ? 36 : 24}
              onDragStart={() => { setDraggingIdx(i); setCursorPos(pt); }}
              onDragMove={(e) => handleCornerDrag(i, e.target.x(), e.target.y())}
              onDragEnd={(e) => {
                handleCornerDrag(i, e.target.x(), e.target.y());
                setDraggingIdx(null);
                setCursorPos(null);
              }}
              onMouseEnter={(e) => {
                (e.target as Konva.Circle).radius(HANDLE_R + 3);
                e.target.getStage()!.container().style.cursor = "grab";
              }}
              onMouseLeave={(e) => {
                (e.target as Konva.Circle).radius(HANDLE_R);
                e.target.getStage()!.container().style.cursor = "crosshair";
              }}
            />
          ))}
        </Layer>
      </Stage>

      {isInteracting && cursorPos && image && (
        <ZoomLens image={image} point={cursorPos} stageSize={stageSize} />
      )}
      {!isInteracting && hasNoRect && hoverPos && image && !IS_TOUCH && (
        <ZoomLens image={image} point={hoverPos} stageSize={stageSize} />
      )}
    </div>
  );
}

// ── Zoom lens that follows cursor position ──

function ZoomLens({
  image, point, stageSize,
}: {
  image: HTMLImageElement;
  point: Point;
  stageSize: { width: number; height: number };
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const sx = image.naturalWidth / stageSize.width;
    const sy = image.naturalHeight / stageSize.height;
    const srcW = (ZOOM_SIZE * sx) / ZOOM_SCALE;
    const srcH = (ZOOM_SIZE * sy) / ZOOM_SCALE;
    const srcX = point.x * sx - srcW / 2;
    const srcY = point.y * sy - srcH / 2;

    ctx.clearRect(0, 0, ZOOM_SIZE, ZOOM_SIZE);
    ctx.drawImage(image, srcX, srcY, srcW, srcH, 0, 0, ZOOM_SIZE, ZOOM_SIZE);

    ctx.strokeStyle = STEGU_RED;
    ctx.lineWidth = 1.5;
    const c = ZOOM_SIZE / 2;
    ctx.beginPath();
    ctx.moveTo(c - 10, c); ctx.lineTo(c + 10, c);
    ctx.moveTo(c, c - 10); ctx.lineTo(c, c + 10);
    ctx.stroke();
  }, [image, point, stageSize]);

  const OFFSET = 20;
  const lensX = point.x + OFFSET + ZOOM_SIZE > stageSize.width
    ? point.x - OFFSET - ZOOM_SIZE
    : point.x + OFFSET;
  const lensY = point.y + OFFSET + ZOOM_SIZE > stageSize.height
    ? point.y - OFFSET - ZOOM_SIZE
    : point.y + OFFSET;

  return (
    <div
      className="absolute pointer-events-none z-20 rounded-xl overflow-hidden shadow-2xl"
      style={{
        width: ZOOM_SIZE, height: ZOOM_SIZE,
        left: Math.max(0, lensX), top: Math.max(0, lensY),
        border: `2px solid ${STEGU_RED}`,
      }}
    >
      <canvas ref={canvasRef} width={ZOOM_SIZE} height={ZOOM_SIZE} className="block" />
      <div className="absolute bottom-1 left-1/2 -translate-x-1/2 px-1.5 py-px rounded bg-black/60 backdrop-blur-sm">
        <span className="text-[8px] font-bold text-white">{ZOOM_SCALE}x</span>
      </div>
    </div>
  );
}
