"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import dynamic from "next/dynamic";
import ImageUpload from "./components/ImageUpload";
import ErrorBoundary from "./components/ErrorBoundary";
import BeforeAfterSlider from "./components/BeforeAfterSlider";

const RectangleDrawer = dynamic(() => import("./components/RectangleDrawer"), {
  ssr: false,
  loading: () => <div className="h-64 sm:h-72 rounded-2xl animate-shimmer" />,
});

interface Point { x: number; y: number }
interface Product {
  productId: string;
  name: string;
  textureImage: string;
  moduleWidthMm: number;
  moduleHeightMm: number;
  jointMm: number;
  layoutType: string;
  offsetRatio: number;
  shopUrl: string;
  category: string;
}
type Stage = "upload" | "edit" | "rendering" | "generated";
interface ScaleInfo {
  wallHeightCm: number;
  wallWidthCm: number;
  courses: number;
  unitsPerRow: number;
  dimensionSource: "calibrated" | "heuristic";
  sceneType: "interior" | "exterior";
  referenceObjects: string[];
}
interface RenderData {
  composite: string | null;
  refined: string | null;
  renderEngine: string;
  timings: Record<string, number>;
  scale?: ScaleInfo;
  analysis?: Record<string, unknown>;
}
type ResultTab = "before" | "after" | "compare";

/* ── Pipeline stage tracking ── */

interface PipelineStage {
  id: string;
  label: string;
  status: "pending" | "running" | "done" | "error" | "warning" | "skipped";
  timing?: number;
  detail?: string;
  error?: string;
  message?: string;
  analysis?: Record<string, unknown>;
}

const INITIAL_STAGES: PipelineStage[] = [
  { id: "mask", label: "Analizowanie sceny", status: "pending" },
  { id: "decode", label: "Przygotowywanie materiałów", status: "pending" },
  { id: "texture", label: "Tworzenie wizualizacji", status: "pending" },
];


/* ── Image preparation (EXIF-safe, size-safe for Vercel) ── */

const CLIENT_MAX_DIM = 2560;
const CLIENT_JPEG_QUALITY = 0.92;
const MAX_DATA_URL_CHARS = 4_500_000;

function getExifOrientation(buffer: ArrayBuffer): number {
  try {
    const view = new DataView(buffer);
    const len = view.byteLength;
    if (len < 14 || view.getUint16(0) !== 0xFFD8) return 1;
    let offset = 2;
    while (offset + 4 <= len) {
      const marker = view.getUint16(offset);
      offset += 2;
      if (marker === 0xFFE1) {
        if (offset + 10 > len) return 1;
        const segLen = view.getUint16(offset);
        if (view.getUint32(offset + 2) !== 0x45786966 || view.getUint16(offset + 6) !== 0x0000) { offset += segLen; continue; }
        const tiffStart = offset + 8;
        if (tiffStart + 8 > len) return 1;
        const le = view.getUint16(tiffStart) === 0x4949;
        const ifd0 = tiffStart + view.getUint32(tiffStart + 4, le);
        if (ifd0 + 2 > len) return 1;
        const count = view.getUint16(ifd0, le);
        for (let i = 0; i < count; i++) {
          const e = ifd0 + 2 + i * 12;
          if (e + 12 > len) return 1;
          if (view.getUint16(e, le) === 0x0112) return view.getUint16(e + 8, le);
        }
        return 1;
      }
      if ((marker & 0xFF00) !== 0xFF00) return 1;
      if (offset + 2 > len) return 1;
      offset += view.getUint16(offset);
    }
  } catch { /* malformed */ }
  return 1;
}

async function prepareImageForUpload(blobUrl: string): Promise<string> {
  const resp = await fetch(blobUrl);
  const blob = await resp.blob();
  const ab = await blob.arrayBuffer();
  const exif = getExifOrientation(ab);
  const bmp = await createImageBitmap(blob);
  const bw = bmp.width, bh = bmp.height;
  const needsCW = exif === 6 && bw > bh;
  const needsCCW = exif === 8 && bw > bh;
  const swaps = needsCW || needsCCW;
  const orientedW = swaps ? bh : bw;
  const orientedH = swaps ? bw : bh;
  const longest = Math.max(orientedW, orientedH);
  const scale = longest > CLIENT_MAX_DIM ? CLIENT_MAX_DIM / longest : 1;
  const tw = Math.round(orientedW * scale);
  const th = Math.round(orientedH * scale);
  const canvas = document.createElement("canvas");
  canvas.width = tw; canvas.height = th;
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (needsCW) { ctx.translate(tw, 0); ctx.rotate(Math.PI / 2); ctx.drawImage(bmp, 0, 0, th, tw); }
  else if (needsCCW) { ctx.translate(0, th); ctx.rotate(-Math.PI / 2); ctx.drawImage(bmp, 0, 0, th, tw); }
  else { ctx.drawImage(bmp, 0, 0, tw, th); }
  bmp.close();
  let quality = CLIENT_JPEG_QUALITY;
  let dataUrl = canvas.toDataURL("image/jpeg", quality);
  while (dataUrl.length > MAX_DATA_URL_CHARS && quality > 0.6) { quality -= 0.08; dataUrl = canvas.toDataURL("image/jpeg", quality); }
  return dataUrl;
}

/* ── Helpers ── */

function formatFastApiError(body: unknown): string {
  if (!body || typeof body !== "object") return "Request failed";
  const d = (body as { detail?: unknown }).detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((e) => typeof e === "object" && e !== null && "msg" in e ? String((e as { msg: unknown }).msg) : JSON.stringify(e)).join("; ");
  if (d !== undefined) return JSON.stringify(d);
  return JSON.stringify(body);
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

function rectToPolygon(pts: Point[]): Point[] {
  if (pts.length < 2) return pts;
  const [a, b] = pts;
  const x0 = Math.min(a.x, b.x), y0 = Math.min(a.y, b.y);
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  return [{ x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }];
}

/* ── Demo photos ── */

const DEMO_ROOMS = [
  { id: "living", label: "Salon", url: "/demo-salon.png" },
  { id: "kitchen", label: "Kuchnia", url: "/demo-kuchnia.png" },
  { id: "exterior", label: "Ściana zewnętrzna", url: "/demo-exterior.png" },
];

/* ── SVG Icons ── */

function IconDownload() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>; }
function IconRefresh() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>; }
function IconClose() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>; }
function IconChevronDown() { return <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2 4l4 4 4-4" strokeLinecap="round" strokeLinejoin="round"/></svg>; }
function IconCheck() { return <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>; }
function IconSparkle() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l2.09 6.26L20 10l-5.91 1.74L12 18l-2.09-6.26L4 10l5.91-1.74L12 2z"/></svg>; }
function IconCart() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 002 1.61h9.72a2 2 0 002-1.61L23 6H6"/></svg>; }

/* ── Loading overlay ── */

const LOADING_MESSAGES = [
  "Analizowanie sceny…",
  "Rozpoznawanie ściany…",
  "Dopasowywanie perspektywy…",
  "Przygotowywanie tekstury…",
  "Tworzenie wizualizacji…",
  "Nakładanie materiału…",
  "Wygładzanie krawędzi…",
  "Finalizowanie…",
];

function AnimatedStatusText({ currentMessage }: { currentMessage?: string }) {
  const [msgIndex, setMsgIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (currentMessage) return;
    const interval = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setMsgIndex(prev => (prev + 1) % LOADING_MESSAGES.length);
        setVisible(true);
      }, 300);
    }, 2200);
    return () => clearInterval(interval);
  }, [currentMessage]);

  const text = currentMessage || LOADING_MESSAGES[msgIndex];

  return (
    <p
      className={`text-xs text-stone-500 font-medium tracking-wide transition-all duration-300 ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-1'
      }`}
    >
      {text}
    </p>
  );
}

function PipelineStatusPanel({ stages, isOverlay }: { stages: PipelineStage[]; isOverlay?: boolean }) {
  const currentRunning = stages.find(s => s.status === "running");
  const completedCount = stages.filter(s => s.status === "done" || s.status === "warning" || s.status === "skipped").length;
  const totalStages = stages.length;
  const hasErrors = stages.some(s => s.status === "error");
  const errorStage = stages.find(s => s.status === "error");
  const progress = Math.round((completedCount / totalStages) * 100);

  if (!isOverlay) {
    // Inline (non-overlay) — just show small progress
    return (
      <div className="w-full flex items-center gap-3 px-2">
        <div className="grid grid-cols-3 gap-[2px] w-6 h-6 shrink-0">
          {Array.from({ length: 9 }).map((_, i) => (
            <div key={i} className="rounded-[1px] grid-loading-cell" style={{ animationDelay: `${i * 0.08}s` }} />
          ))}
        </div>
        <div className="flex-1">
          <div className="w-full h-1 bg-stone-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-[#A01B1B] rounded-full transition-all duration-700 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
        <span className="text-[10px] font-medium text-stone-400 tabular-nums shrink-0">{progress}%</span>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-white/95 backdrop-blur-2xl rounded-2xl animate-fade-in p-8">
      {/* Grid loading animation */}
      <div className="mb-8">
        <div className="grid grid-cols-3 gap-[4px] w-16 h-16">
          {Array.from({ length: 9 }).map((_, i) => (
            <div
              key={i}
              className="rounded-[3px] grid-loading-cell"
              style={{ animationDelay: `${i * 0.1}s` }}
            />
          ))}
        </div>
      </div>

      {/* Progress bar — wide and smooth */}
      <div className="w-48 mb-5">
        <div className="w-full h-[3px] bg-stone-100 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${progress}%`,
              background: 'linear-gradient(90deg, #A01B1B, #c93434)',
            }}
          />
        </div>
      </div>

      {/* Animated status text */}
      {hasErrors ? (
        <p className="text-sm text-red-500 font-medium text-center px-4">
          {errorStage?.error || "Wystąpił błąd. Spróbuj ponownie."}
        </p>
      ) : (
        <div className="h-6 flex items-center">
          <AnimatedStatusText currentMessage={currentRunning?.message || currentRunning?.label} />
        </div>
      )}
    </div>
  );
}

/* ── Step indicator ── */

const STEPS = [
  { n: 1, label: "Zdjęcie" },
  { n: 2, label: "Ściana i materiał" },
  { n: 3, label: "Wizualizacja" },
  { n: 4, label: "Gotowe" },
];

function StepBar({ step }: { step: number }) {
  return (
    <div className="bg-white border-b border-stone-200/60">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-3 sm:py-4">
        <div className="flex items-center">
          {STEPS.map(({ n, label }, i) => (
            <div key={n} className="flex items-center flex-1 last:flex-none">
              <div className="flex flex-col items-center gap-1.5 shrink-0">
                <span className={`w-7 h-7 sm:w-8 sm:h-8 rounded-full flex items-center justify-center text-[11px] sm:text-xs font-bold transition-all ${
                  n < step
                    ? "bg-[#A01B1B] text-white"
                    : n === step
                      ? "border-2 border-[#A01B1B] text-[#A01B1B] bg-white"
                      : "border border-stone-200 text-stone-400 bg-stone-50"
                }`}>
                  {n < step ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg> : n}
                </span>
                <span className={`text-[10px] sm:text-[11px] font-medium whitespace-nowrap ${
                  n < step ? "text-[#A01B1B]" : n === step ? "text-stone-800" : "text-stone-400"
                }`}>
                  {label}
                </span>
              </div>
              {i < STEPS.length - 1 && (
                <div className={`flex-1 h-[2px] mx-2 sm:mx-3 rounded-full transition-colors -mt-5 ${
                  n < step ? "bg-[#A01B1B]" : "bg-stone-200"
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── Product card ── */

function ProductCard({ product, active, onSelect }: { product: Product; active: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`group relative rounded-xl overflow-hidden transition-all cursor-pointer text-left ${
        active
          ? "ring-2 ring-[#A01B1B] ring-offset-2 ring-offset-white shadow-md scale-[1.02]"
          : "ring-1 ring-stone-200 hover:ring-stone-300 hover:shadow-sm"
      }`}
    >
      <img src={`/api/textures/${product.productId}`} alt={product.name} className="aspect-square w-full object-cover" />
      <div className={`absolute inset-x-0 bottom-0 px-2 py-1.5 ${
        active ? "bg-[#7A1515]/90" : "bg-gradient-to-t from-black/70 via-black/30 to-transparent"
      }`}>
        <p className="text-[10px] sm:text-[11px] font-medium text-white leading-tight truncate">{product.name}</p>
        <p className="text-[8px] sm:text-[9px] text-white/60 mt-0.5">
          {product.moduleWidthMm}×{product.moduleHeightMm}mm
        </p>
      </div>
      {active && (
        <div className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-[#A01B1B] flex items-center justify-center shadow-sm">
          <IconCheck />
        </div>
      )}
    </button>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════
   Main component
   ══════════════════════════════════════════════════════════════════════════════ */

export default function Home() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [originalImageSrc, setOriginalImageSrc] = useState<string | null>(null);
  const [rectPts, setRectPts] = useState<Point[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [productsLoading, setProductsLoading] = useState(true);
  const prevUrl = useRef<string | null>(null);
  const [stageSize, setStageSizeState] = useState<{ width: number; height: number } | null>(null);
  const [stage, setStage] = useState<Stage>("upload");
  const [renderData, setRenderData] = useState<RenderData | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [genProgress, setGenProgress] = useState("");
  const [resultTab, setResultTab] = useState<ResultTab>("compare");
  const [mobileProductOpen, setMobileProductOpen] = useState(false);
  const [remaining, setRemaining] = useState<{ remaining: number; limit: number; unlimited: boolean } | null>(null);
  const [demoLoading, setDemoLoading] = useState<string | null>(null);
  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>(INITIAL_STAGES);

  // ── Refine (Gemini photorealistic render) ──
  const [refining, setRefining] = useState(false);
  const [refinedImage, setRefinedImage] = useState<string | null>(null);
  const [refineError, setRefineError] = useState<string | null>(null);
  const [refineMsg, setRefineMsg] = useState("");
  const refineIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const REFINE_MESSAGES = [
    "Analizowanie oświetlenia sceny…",
    "Dopasowywanie cieni i odbić…",
    "Wygładzanie krawędzi tekstury…",
    "Korygowanie perspektywy…",
    "Dodawanie realistycznych detali…",
    "Optymalizowanie fotorealizmu…",
    "Ulepszanie przejść kolorystycznych…",
    "Finalizowanie renderingu…",
  ];

  const handleRefine = useCallback(async () => {
    if (!renderData?.refined || !selectedProduct) return;
    setRefining(true);
    setRefineError(null);
    setRefinedImage(null);
    let msgIdx = 0;
    setRefineMsg(REFINE_MESSAGES[0]);
    refineIntervalRef.current = setInterval(() => {
      msgIdx = (msgIdx + 1) % REFINE_MESSAGES.length;
      setRefineMsg(REFINE_MESSAGES[msgIdx]);
    }, 5000);

    try {
      // Send directly to Python API (bypass Next.js proxy — body too large)
      const resp = await fetch(`${API_BASE}/api/refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          composite: renderData.refined,
          original: originalImageSrc || imageSrc,
          product_id: selectedProduct.productId,
          product_name: selectedProduct.name,
          material_type: selectedProduct.layoutType,
        }),
      });

      if (!resp.ok) {
        let errorMsg = "Nie udało się wygenerować renderingu";
        try {
          const errData = await resp.json();
          if (errData.detail) errorMsg = errData.detail;
        } catch { /* use default */ }
        throw new Error(errorMsg);
      }

      const data = await resp.json();
      if (!data.image) throw new Error("Model nie wygenerował obrazu — spróbuj ponownie");
      setRefinedImage(data.image);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Nieznany błąd — spróbuj ponownie";
      setRefineError(msg);
    } finally {
      setRefining(false);
      if (refineIntervalRef.current) clearInterval(refineIntervalRef.current);
    }
  }, [renderData, selectedProduct, originalImageSrc, imageSrc]);


  const fetchRemaining = useCallback(() => {
    fetch(`${API_BASE}/api/remaining-generations`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setRemaining({ remaining: d.remaining, limit: d.limit, unlimited: d.unlimited }))
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch("/api/products")
      .then(r => r.ok ? r.json() : [])
      .then((d: Product[]) => setProducts(d))
      .catch(() => {})
      .finally(() => setProductsLoading(false));
    fetchRemaining();
  }, [fetchRemaining]);

  const handleImageSelected = useCallback((objectUrl: string) => {
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    prevUrl.current = objectUrl;
    setImageSrc(objectUrl);
    if (!originalImageSrc) setOriginalImageSrc(objectUrl);
    setRectPts([]);
    setStage("edit");
    setRenderData(null);
    setPipelineError(null);
    setGenProgress("");
  }, [originalImageSrc]);

  useEffect(() => { return () => { if (prevUrl.current) URL.revokeObjectURL(prevUrl.current); }; }, []);

  const handleDemoRoom = useCallback(async (url: string, id: string) => {
    setDemoLoading(id);
    try {
      const resp = await fetch(url);
      const blob = await resp.blob();
      const objectUrl = URL.createObjectURL(blob);
      handleImageSelected(objectUrl);
    } catch {
      /* network error — fail silently */
    } finally {
      setDemoLoading(null);
    }
  }, [handleImageSelected]);

  function resetAll() {
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    prevUrl.current = null;
    setImageSrc(null);
    setOriginalImageSrc(null);
    setRectPts([]);
    setSelectedProduct(null);
    setStage("upload");
    setRenderData(null);
    setPipelineError(null);
    setGenProgress("");
  }

  function resetToEdit() {
    setRectPts([]);
    setStage("edit");
    setRenderData(null);
    setPipelineError(null);
    setGenProgress("");
  }

  const handleStageSizeChange = useCallback((size: { width: number; height: number }) => setStageSizeState(size), []);

  const wallDefined = rectPts.length >= 2;
  const canGenerate = wallDefined && imageSrc !== null && selectedProduct !== null;

  const fetchWithRetry = useCallback(async (url: string, opts: RequestInit, maxAttempts = 3): Promise<Response> => {
    let last: Error | null = null;
    for (let i = 1; i <= maxAttempts; i++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 270_000);
      try {
        const res = await fetch(url, { ...opts, signal: controller.signal });
        clearTimeout(timer);
        if (res.ok || res.status < 500) return res;
        if (i < maxAttempts) { setGenProgress(`Ponawianie (${i}/${maxAttempts})…`); await new Promise(r => setTimeout(r, 2000 * i)); continue; }
        return res;
      } catch (e) {
        clearTimeout(timer);
        if (e instanceof DOMException && e.name === "AbortError") { last = new Error("Przekroczono limit czasu — spróbuj ponownie"); break; }
        last = e instanceof Error ? e : new Error(String(e));
        if (i < maxAttempts) { setGenProgress(`Błąd sieci — ponawianie (${i}/${maxAttempts})…`); await new Promise(r => setTimeout(r, 2000 * i)); }
      }
    }
    throw last ?? new Error("Request failed");
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!imageSrc || !canGenerate || !stageSize || !selectedProduct) return;
    setStage("rendering");
    setPipelineError(null);
    setRenderData(null);
    setRefinedImage(null);
    setRefineError(null);
    setGenProgress("Przygotowywanie obrazu…");
    setPipelineStages(INITIAL_STAGES.map(s => ({ ...s, status: "pending" as const })));


    try {
      const imageBase64 = await prepareImageForUpload(imageSrc);
      const polygon = rectToPolygon(rectPts);

      // ── Stage 1: Smart wall mask via CV Engine ──
      setPipelineStages(prev => prev.map(s => s.id === "mask" ? { ...s, status: "running", message: "Analizowanie ściany…" } : s));
      setGenProgress("Analizowanie ściany…");

      let confirmedMask: string | null = null;
      let calibrationData: Record<string, unknown> | null = null;
      try {
        const maskRes = await fetch(`${API_BASE}/api/smart-mask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: imageBase64,
            polygon,
            canvas_width: stageSize.width,
            canvas_height: stageSize.height,
          }),
        });

        if (maskRes.ok) {
          const maskData = await maskRes.json();
          confirmedMask = maskData.final_mask || maskData.wall_mask;
          calibrationData = maskData.calibration || null;
          const timing = maskData.timings?.total;
          setPipelineStages(prev => prev.map(s => s.id === "mask"
            ? { ...s, status: "done", timing }
            : s
          ));
        } else {
          // Fallback to polygon-only if cv-engine unavailable
          setPipelineStages(prev => prev.map(s => s.id === "mask"
            ? { ...s, status: "warning", detail: "Fallback: polygon-only" }
            : s
          ));
        }
      } catch {
        // cv-engine not running — fallback gracefully
        setPipelineStages(prev => prev.map(s => s.id === "mask"
          ? { ...s, status: "warning", detail: "CV Engine niedostępny — użyto prostej maski" }
          : s
        ));
      }

      // ── Stage 2+3: Render stream (decode + texture) ──
      setGenProgress("Łączenie z serwerem…");

      const res = await fetch(`${API_BASE}/api/render-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageBase64,
          polygon,
          confirmed_mask: confirmedMask,
          calibration: calibrationData,
          product_id: selectedProduct.productId,
          canvas_width: stageSize.width,
          canvas_height: stageSize.height,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(formatFastApiError(err) || `Server error ${res.status}`);
      }

      // Read SSE stream
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");
      const decoder = new TextDecoder();
      let buffer = "";
      let finalAnalysis: Record<string, unknown> | undefined;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;
          let evt: Record<string, unknown>;
          try {
            evt = JSON.parse(line.slice(6));
          } catch { continue; }

          const stageId = evt.stage as string;
          const status = evt.status as PipelineStage["status"] | undefined;

          // Handle final "done" event
          if (stageId === "done") {
            const result = evt.result as Record<string, unknown> | undefined;
            if (result) {
              setRenderData({
                composite: (result.composite as string) || null,
                refined: (result.refined as string) || (result.composite as string) || null,
                renderEngine: "deterministic",
                timings: {},
                analysis: (result.analysis as Record<string, unknown>) || finalAnalysis,
              });
              setResultTab("compare");
              setStage("generated");
              fetchRemaining();
            } else if (evt.ok === false) {
              throw new Error((evt.error as string) || "Pipeline failed");
            }
            continue;
          }

          // Handle mask_refine (not in the main list)
          if (stageId === "mask_refine") continue;

          // Save analysis data if present
          if (evt.analysis) {
            finalAnalysis = evt.analysis as Record<string, unknown>;
          }

          // Update stage status
          if (status) {
            setPipelineStages(prev => prev.map(s =>
              s.id === stageId
                ? {
                    ...s,
                    status,
                    timing: evt.timing as number | undefined,
                    detail: evt.detail as string | undefined,
                    error: evt.error as string | undefined,
                    message: evt.message as string | undefined,
                    analysis: evt.analysis as Record<string, unknown> | undefined ?? s.analysis,
                  }
                : s
            ));
            // Update progress text
            if (status === "running" && evt.message) {
              setGenProgress(evt.message as string);
            }
          }
        }
      }
    } catch (e) {
      setPipelineError(e instanceof Error ? e.message : "Generowanie nie powiodło się");
      setStage("edit");
    } finally {
      setGenProgress("");
    }
  }, [imageSrc, canGenerate, selectedProduct, stageSize, rectPts, fetchRemaining]);

  const handleUseResultAsInput = useCallback(() => {
    if (!renderData?.refined) return;
    const b64 = renderData.refined;
    const byteString = atob(b64.split(",")[1]);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
    const blob = new Blob([ab], { type: "image/jpeg" });
    const url = URL.createObjectURL(blob);
    handleImageSelected(url);
  }, [renderData, handleImageSelected]);

  const handleDownload = useCallback((url: string, suffix: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = `stegu-${selectedProduct?.productId ?? "result"}-${suffix}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [selectedProduct]);

  const step = stage === "upload" ? 1 : stage === "edit" ? 2 : stage === "rendering" ? 3 : 4;

  /* ── Category icons ── */
  const CategoryIcon = ({ cat, className = "" }: { cat: string; className?: string }) => {
    const cls = `w-4 h-4 ${className}`;
    switch (cat) {
      case "cegły":
        return (
          <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="4" width="20" height="5" rx="1" /><rect x="2" y="11" width="9" height="5" rx="1" /><rect x="13" y="11" width="9" height="5" rx="1" /><rect x="6" y="18" width="14" height="3" rx="1" />
          </svg>
        );
      case "kamień":
        return (
          <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 8c0-1 1-3 4-3s3 2 5 2 3-2 5-2 4 1 4 3v7c0 2-2 4-4 4H8c-2 0-4-2-4-4V8z" />
            <path d="M8 13h3M14 10h2" />
          </svg>
        );
      case "lamele":
        return (
          <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="4" height="18" rx="1" /><rect x="10" y="3" width="4" height="18" rx="1" /><rect x="17" y="3" width="4" height="18" rx="1" />
          </svg>
        );
      default:
        return (
          <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 21V9" />
          </svg>
        );
    }
  };

  /* ── Product grid (reused in sidebar + mobile sheet) ── */

  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({});

  const toggleCategory = useCallback((cat: string) => {
    setOpenCategories(prev => ({ ...prev, [cat]: !prev[cat] }));
  }, []);

  // Group products by category
  const grouped = products.reduce<Record<string, Product[]>>((acc, p) => {
    const cat = p.category || "inne";
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(p);
    return acc;
  }, {});

  // Ensure all categories start as open
  useEffect(() => {
    if (products.length > 0 && Object.keys(openCategories).length === 0) {
      const initial: Record<string, boolean> = {};
      Object.keys(grouped).forEach(cat => { initial[cat] = false; });
      setOpenCategories(initial);
    }
  }, [products.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const CATEGORY_ORDER = ["cegły", "kamień", "lamele", "inne"];

  const ProductGrid = ({ onPick }: { onPick?: () => void }) => (
    <>
      {productsLoading ? (
        <div className="space-y-3">
          {[0,1].map(i => (
            <div key={i} className="rounded-xl border border-stone-100 p-3">
              <div className="h-4 w-24 rounded animate-shimmer mb-2" />
              <div className="grid grid-cols-3 sm:grid-cols-2 gap-2">
                {[0,1].map(j => <div key={j} className="aspect-square rounded-lg animate-shimmer" />)}
              </div>
            </div>
          ))}
        </div>
      ) : products.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-10 text-center">
          <p className="text-sm text-stone-400">Brak dostępnych materiałów</p>
          <p className="text-xs text-stone-300 mt-1">Dodaj produkty w panelu administracyjnym</p>
        </div>
      ) : (
        <div className="space-y-2">
          {CATEGORY_ORDER.filter(cat => grouped[cat]).map(cat => {
            const prods = grouped[cat];
            const isOpen = openCategories[cat] !== false;
            const hasSelected = prods.some(p => selectedProduct?.productId === p.productId);

            return (
              <div key={cat} className={`rounded-xl border transition-colors ${
                hasSelected ? "border-[#A01B1B]/20 bg-[#A01B1B]/[0.02]" : "border-stone-100 bg-white"
              }`}>
                {/* Category header */}
                <button
                  type="button"
                  onClick={() => toggleCategory(cat)}
                  className="w-full flex items-center gap-2.5 px-3 py-2.5 cursor-pointer hover:bg-stone-50/50 rounded-xl transition-colors"
                >
                  <div className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 ${
                    hasSelected ? "bg-[#A01B1B]/10" : "bg-stone-100"
                  }`}>
                    <CategoryIcon cat={cat} className={hasSelected ? "text-[#A01B1B]" : "text-stone-400"} />
                  </div>
                  <div className="flex-1 text-left">
                    <p className={`text-[11px] font-semibold leading-tight capitalize ${
                      hasSelected ? "text-[#A01B1B]" : "text-stone-600"
                    }`}>{cat}</p>
                    <p className="text-[9px] text-stone-400 mt-0.5">{prods.length} {prods.length === 1 ? "produkt" : prods.length < 5 ? "produkty" : "produktów"}</p>
                  </div>
                  <svg
                    className={`w-3.5 h-3.5 text-stone-300 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
                    viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2"
                  >
                    <path d="M2 4l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>

                {/* Products */}
                {isOpen && (
                  <div className="px-2.5 pb-2.5">
                    <div className="grid grid-cols-3 sm:grid-cols-2 gap-2">
                      {prods.map(p => (
                        <ProductCard
                          key={p.productId}
                          product={p}
                          active={selectedProduct?.productId === p.productId}
                          onSelect={() => { setSelectedProduct(p); onPick?.(); }}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </>
  );

  /* ══════════════════════════════════════════════════════════════════════════
     Render
     ══════════════════════════════════════════════════════════════════════════ */

  return (
    <div className="h-[100svh] flex flex-col bg-[#faf9f7] overflow-hidden">
      {/* ── Header ── */}
      <header className="stegu-gradient shrink-0 z-40">
        <div className="max-w-[1440px] mx-auto px-3 sm:px-6 lg:px-8 h-12 sm:h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <img src="/stegu-logo.png" alt="Stegu" className="h-6 sm:h-7" />
            <div className="w-px h-5 bg-stone-300" />
            <span className="text-stone-500 text-[10px] sm:text-[11px] font-medium tracking-widest uppercase">Visualizer</span>
          </div>
          <div className="flex items-center gap-3">
            {remaining && !remaining.unlimited && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-stone-200/60">
                <span className={`text-[10px] sm:text-[11px] font-semibold ${remaining.remaining <= 3 ? "text-red-600" : "text-stone-600"}`}>
                  {remaining.remaining}/{remaining.limit}
                </span>
                <span className="text-[9px] text-stone-400 hidden sm:inline">generacji</span>
              </div>
            )}
            {stage !== "upload" && (
              <button type="button" onClick={resetAll} className="text-[11px] text-stone-500 hover:text-stone-800 transition-colors cursor-pointer">
                Zacznij od nowa
              </button>
            )}
          </div>
        </div>
      </header>

      <StepBar step={step} />

      {/* ── Main ── */}
      <main className="flex-1 min-h-0 overflow-y-auto">
      <div className="max-w-[1440px] w-full mx-auto px-3 sm:px-6 lg:px-8 py-3 sm:py-5">

        {/* ═══ STAGE: Upload / Hero ═══ */}
        {stage === "upload" && (
          <div className="animate-fade-in-up max-w-2xl mx-auto flex flex-col">
            {/* Hero */}
            <div className="text-center mb-3 sm:mb-5">
              <h1 className="text-2xl sm:text-3xl font-semibold text-stone-800 tracking-tight font-[family-name:var(--font-outfit)]">
                Zobacz produkt na swojej ścianie
              </h1>
              <p className="text-stone-500 mt-1.5 text-xs sm:text-sm max-w-lg mx-auto leading-relaxed hidden sm:block">
                Wgraj swoje zdjęcie, zaznacz ścianę i sprawdź jak będzie wyglądał wybrany produkt Stegu — w realistycznej wizualizacji.
              </p>
              <div className="flex items-center justify-center gap-4 mt-2 text-[10px] text-stone-400">
                <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#A01B1B]/40" /> Bezpłatne</span>
                <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#A01B1B]/40" /> Bez rejestracji</span>
                <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#A01B1B]/40" /> Wynik w ~15s</span>
              </div>
            </div>

            {/* Upload area */}
            <ImageUpload onImageSelected={handleImageSelected} hasImage={false} />

            {/* Demo rooms */}
            <div className="mt-3 sm:mt-5">
              <div className="flex items-center gap-2 mb-2 sm:mb-3">
                <div className="h-px flex-1 bg-stone-200" />
                <span className="text-[10px] text-stone-400 font-medium px-2 whitespace-nowrap">lub wypróbuj przykładowe zdjęcie</span>
                <div className="h-px flex-1 bg-stone-200" />
              </div>
              <div className="grid grid-cols-3 gap-2">
                {DEMO_ROOMS.map(room => (
                  <button
                    key={room.id}
                    type="button"
                    onClick={() => handleDemoRoom(room.url, room.id)}
                    disabled={demoLoading !== null}
                    className="group relative rounded-xl overflow-hidden cursor-pointer aspect-[4/3] border border-stone-200 hover:border-stone-300 transition-all hover:shadow-md disabled:opacity-60"
                  >
                    <img src={room.url} alt={room.label} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" loading="lazy" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                    <span className="absolute bottom-1.5 left-2 text-[10px] font-semibold text-white drop-shadow-sm">
                      {demoLoading === room.id ? "Ładowanie…" : room.label}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Trust signals — hidden on mobile to avoid scroll */}
            <div className="mt-4 hidden sm:grid grid-cols-3 gap-2 sm:gap-3">
              {[
                { title: "Wgraj zdjęcie", desc: "Własne lub z przykładów", num: "1" },
                { title: "Zaznacz ścianę", desc: "Wybierz produkt Stegu", num: "2" },
                { title: "Zamów", desc: "Wizualizacja + link do sklepu", num: "3" },
              ].map(({ title, desc, num }) => (
                <div key={num} className="flex gap-2 items-start p-3 rounded-xl bg-white border border-stone-200/80">
                  <div className="w-6 h-6 rounded-md bg-[#A01B1B]/10 flex items-center justify-center shrink-0 mt-0.5">
                    <span className="text-[10px] font-bold text-[#A01B1B]">{num}</span>
                  </div>
                  <div>
                    <p className="text-[11px] font-semibold text-stone-700">{title}</p>
                    <p className="text-[10px] text-stone-400 leading-snug">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ═══ STAGE: Edit / Render / Generated ═══ */}
        {stage !== "upload" && (
          <div className="flex flex-col lg:flex-row gap-3 sm:gap-4 animate-fade-in">
            {/* Left: canvas / result */}
            <div className="flex-1 min-w-0 min-h-0">
              <div className="bg-white rounded-2xl border border-stone-200/80 shadow-sm overflow-hidden flex flex-col">
                {/* Panel header */}
                <div className="px-4 sm:px-5 py-3 border-b border-stone-100 flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <h2 className="text-sm font-semibold text-stone-700">
                      {stage === "edit" ? "Zaznacz ścianę" : stage === "rendering" ? "Generowanie wizualizacji…" : "Twoja wizualizacja"}
                    </h2>
                    {stage === "edit" && wallDefined && (
                      <span className="px-2 py-0.5 rounded-full bg-[#A01B1B]/10 text-[#A01B1B] text-[10px] font-semibold">
                        Ściana zaznaczona
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {stage === "edit" && (
                      <button type="button" onClick={() => { setImageSrc(null); setStage("upload"); }} className="text-[11px] text-stone-400 hover:text-stone-600 cursor-pointer transition-colors">
                        Zmień zdjęcie
                      </button>
                    )}
                    {stage === "edit" && (
                      <button
                        type="button"
                        onClick={() => setMobileProductOpen(true)}
                        className="lg:hidden flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold text-[#A01B1B] bg-red-50 rounded-lg border border-red-200 cursor-pointer"
                      >
                        {selectedProduct ? selectedProduct.name : "Wybierz materiał"}
                        <IconChevronDown />
                      </button>
                    )}
                  </div>
                </div>

                {/* Canvas area */}
                <div className="relative flex-1 min-h-0 overflow-y-auto p-2 sm:p-3" style={{ minHeight: "200px" }}>
                  {stage === "edit" && imageSrc && (
                    <ErrorBoundary>
                      <p className="text-[11px] text-stone-400 mb-2.5">
                        Narysuj prostokąt na ścianie — przeciągaj narożniki aby dopasować
                      </p>
                      <RectangleDrawer
                        imageSrc={imageSrc}
                        points={rectPts}
                        onPointsChange={setRectPts}
                        onStageSizeChange={handleStageSizeChange}
                      />
                    </ErrorBoundary>
                  )}

                  {stage === "rendering" && <PipelineStatusPanel stages={pipelineStages} isOverlay />}

                  {stage === "generated" && renderData && (
                    <div className="flex flex-col gap-3 sm:gap-4 animate-scale-in">
                      {/* Result tabs */}
                      <div className="border-b border-stone-200 flex">
                        {([["compare", "Porównanie"], ["after", "Po"], ["before", "Przed"]] as [ResultTab, string][]).map(([tab, label]) => (
                          <button
                            key={tab}
                            type="button"
                            onClick={() => setResultTab(tab)}
                            className={`px-3 sm:px-4 py-2 text-[11px] sm:text-xs font-semibold border-b-2 -mb-px transition-colors cursor-pointer whitespace-nowrap ${
                              resultTab === tab ? "border-[#A01B1B] text-stone-800" : "border-transparent text-stone-400 hover:text-stone-600"
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>

                      {/* Result content */}
                      {resultTab === "compare" && (originalImageSrc || imageSrc) && renderData.refined ? (
                        <BeforeAfterSlider before={originalImageSrc || imageSrc!} after={renderData.refined} initialPosition={50} />
                      ) : resultTab === "after" && renderData.refined ? (
                        <img src={renderData.refined} alt="Po wizualizacji" className="rounded-xl w-full" />
                      ) : resultTab === "before" && (originalImageSrc || imageSrc) ? (
                        <img src={originalImageSrc || imageSrc!} alt="Przed wizualizacją" className="rounded-xl w-full" />
                      ) : (
                        <div className="p-8 rounded-xl bg-stone-50 text-center">
                          <p className="text-sm text-stone-400">Brak danych</p>
                        </div>
                      )}

                      {/* Refine button / loading / result */}
                      {refinedImage ? (
                        <div className="flex flex-col gap-3">
                          <div className="flex items-center gap-2 px-1">
                            <div className="w-2 h-2 rounded-full bg-emerald-500" />
                            <p className="text-xs font-medium text-emerald-700">Fotorealistyczny render gotowy</p>
                          </div>
                          {resultTab === "compare" && (originalImageSrc || imageSrc) ? (
                            <BeforeAfterSlider before={originalImageSrc || imageSrc!} after={refinedImage} initialPosition={50} />
                          ) : resultTab === "before" && (originalImageSrc || imageSrc) ? (
                            <img src={originalImageSrc || imageSrc!} alt="Oryginał" className="rounded-xl w-full" />
                          ) : (
                            <img src={refinedImage} alt="Fotorealistyczny render" className="rounded-xl w-full" />
                          )}
                        </div>
                      ) : refining ? (
                        <div className="flex flex-col items-center gap-4 py-8 px-4">
                          <div className="relative w-12 h-12">
                            <div className="absolute inset-0 rounded-full border-2 border-stone-200" />
                            <div className="absolute inset-0 rounded-full border-2 border-t-[#A01B1B] animate-spin" />
                          </div>
                          <div className="text-center">
                            <p className="text-sm font-medium text-stone-700 animate-pulse">{refineMsg}</p>
                            <p className="text-[10px] text-stone-400 mt-1">Szacowany czas: ~45 sekund</p>
                          </div>
                          <div className="w-full max-w-xs bg-stone-100 rounded-full h-1.5 overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-[#A01B1B] to-[#c42b2b] rounded-full animate-progress" />
                          </div>
                        </div>
                      ) : (
                        <button
                          type="button"
                          onClick={handleRefine}
                          className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-stone-800 to-stone-900 text-white text-sm font-semibold hover:from-stone-700 hover:to-stone-800 transition-all duration-200 shadow-lg hover:shadow-xl cursor-pointer group"
                        >
                          <svg className="w-4 h-4 text-amber-400 group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
                          </svg>
                          Fotorealistyczny render
                        </button>
                      )}

                      {refineError && (
                        <p className="text-xs text-red-500 px-1">{refineError}</p>
                      )}

                      {/* Product info */}
                      {selectedProduct && (
                        <div className="flex items-start gap-3 px-1">
                          <img src={`/api/textures/${selectedProduct.productId}`} alt="" className="w-10 h-10 rounded-lg object-cover border border-stone-200 mt-0.5" />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-semibold text-stone-700">{selectedProduct.name}</p>
                            <p className="text-[10px] text-stone-400">
                              {selectedProduct.moduleWidthMm}×{selectedProduct.moduleHeightMm}mm
                            </p>
                          </div>
                        </div>
                      )}


                    </div>
                  )}
                </div>

                {/* Bottom bar */}
                <div className="px-3 sm:px-5 py-3 border-t border-stone-100 flex flex-wrap items-center gap-2 sm:gap-3 bg-stone-50/50">
                  {/* Edit stage CTAs */}
                  {stage === "edit" && canGenerate && (
                    <button
                      type="button"
                      onClick={handleGenerate}
                      className="px-5 sm:px-7 py-2.5 text-xs sm:text-sm font-semibold text-white bg-[#A01B1B] rounded-xl hover:bg-[#8A1717] shadow-sm transition-all hover:shadow-md cursor-pointer flex items-center gap-2 flex-1 sm:flex-none justify-center"
                    >
                      <IconSparkle />
                      Generuj wizualizację
                    </button>
                  )}
                  {stage === "edit" && wallDefined && !selectedProduct && (
                    <p className="text-[11px] text-stone-400">
                      <span className="hidden lg:inline">Teraz wybierz materiał z panelu po prawej →</span>
                      <span className="lg:hidden">Kliknij &quot;Wybierz materiał&quot; powyżej</span>
                    </p>
                  )}
                  {stage === "edit" && !wallDefined && (
                    <p className="text-[11px] text-stone-400">
                      Narysuj prostokąt na zdjęciu aby zaznaczyć ścianę
                    </p>
                  )}

                  {/* Generated stage CTAs */}
                  {stage === "generated" && (
                    <div className="flex flex-col gap-3 w-full">
                      {/* Primary row */}
                      <div className="flex flex-wrap items-center gap-2">
                        {selectedProduct && (
                          <a
                            href={selectedProduct.shopUrl || "https://stegu.pl/produkty/plytki-ceglopodobne/"}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="px-5 py-2.5 text-xs sm:text-sm font-semibold text-white bg-[#A01B1B] rounded-xl hover:bg-[#8A1717] shadow-sm transition-all cursor-pointer flex items-center gap-2"
                          >
                            <IconCart />
                            <span className="hidden sm:inline">Podoba się? Zamów produkt</span>
                            <span className="sm:hidden">Zamów produkt</span>
                          </a>
                        )}
                        {renderData?.refined && (
                          <button
                            type="button"
                            onClick={() => handleDownload(renderData.refined!, "final")}
                            className="px-4 py-2.5 text-xs sm:text-sm font-medium text-stone-700 border border-stone-300 rounded-xl hover:bg-stone-50 transition-all cursor-pointer flex items-center gap-2"
                          >
                            <IconDownload />
                            Pobierz wizualizację
                          </button>
                        )}
                      </div>
                      {/* Secondary row */}
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          onClick={handleUseResultAsInput}
                          className="px-4 py-2 text-xs font-medium text-stone-500 border border-stone-200 rounded-xl hover:bg-stone-50 transition-all cursor-pointer flex items-center gap-2"
                        >
                          <IconRefresh />
                          Dodaj kolejny materiał
                        </button>
                        <button type="button" onClick={resetToEdit} className="px-3 py-2 text-xs text-stone-400 hover:text-stone-600 cursor-pointer transition-colors">
                          Zmień zaznaczenie
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Error */}
                  {pipelineError && (
                    <div className="flex items-center gap-3 w-full bg-red-50 border border-red-200 rounded-xl px-4 py-3 animate-fade-in">
                      <span className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
                      <p className="text-xs text-red-700 flex-1">{pipelineError}</p>
                      {canGenerate && (
                        <button
                          type="button"
                          onClick={() => { setPipelineError(null); handleGenerate(); }}
                          className="text-xs font-semibold text-red-700 underline underline-offset-2 cursor-pointer shrink-0"
                        >
                          Spróbuj ponownie
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right: product sidebar (desktop) */}
            <div className="hidden lg:flex flex-col w-64 xl:w-72 shrink-0 min-h-0">
              <div className="bg-white rounded-2xl border border-stone-200/80 shadow-sm flex flex-col min-h-0 flex-1">
                <div className="px-4 py-4 border-b border-stone-100">
                  <h3 className="text-sm font-semibold text-stone-700">Materiały dekoracyjne</h3>
                  <p className="text-[11px] text-stone-400 mt-0.5">Wybierz produkt do wizualizacji</p>
                  {selectedProduct && (
                    <div className="flex items-center gap-2 mt-2.5 px-2.5 py-2 bg-[#A01B1B]/5 rounded-lg border border-[#A01B1B]/15">
                      <img src={`/api/textures/${selectedProduct.productId}`} alt="" className="w-8 h-8 rounded-md object-cover" />
                      <div>
                        <p className="text-xs font-semibold text-[#A01B1B]">{selectedProduct.name}</p>
                        <p className="text-[10px] text-stone-400">{selectedProduct.moduleWidthMm}×{selectedProduct.moduleHeightMm}mm</p>
                      </div>
                    </div>
                  )}
                </div>
                <div className="p-3 flex-1 overflow-y-auto scrollbar-thin min-h-0">
                  <ProductGrid />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      </main>

      {/* ── Mobile product sheet ── */}
      {mobileProductOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setMobileProductOpen(false)} />
          <div className="absolute bottom-0 inset-x-0 bg-white rounded-t-3xl shadow-2xl animate-slide-up max-h-[75vh] flex flex-col">
            <div className="flex items-center justify-between px-5 py-4 border-b border-stone-100">
              <div>
                <h3 className="text-sm font-semibold text-stone-700">Materiały dekoracyjne</h3>
                <p className="text-[11px] text-stone-400">Wybierz produkt do wizualizacji</p>
              </div>
              <button onClick={() => setMobileProductOpen(false)} className="w-9 h-9 rounded-full bg-stone-100 flex items-center justify-center cursor-pointer hover:bg-stone-200 transition-colors">
                <IconClose />
              </button>
            </div>
            <div className="p-4 overflow-y-auto flex-1 scrollbar-thin">
              <ProductGrid onPick={() => setMobileProductOpen(false)} />
            </div>
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <footer className="border-t border-stone-200/60 py-2 sm:py-2.5 bg-white shrink-0">
        <div className="max-w-[1440px] mx-auto px-3 sm:px-6 lg:px-8 flex items-center justify-between">
          <p className="text-[9px] text-stone-400 tracking-wide">
            Stegu Visualizer · Kamień dekoracyjny i cegła
          </p>
          <a href="/admin" className="text-[9px] text-stone-300 hover:text-stone-500 transition-colors">
            Panel administracyjny
          </a>
        </div>
      </footer>
    </div>
  );
}
