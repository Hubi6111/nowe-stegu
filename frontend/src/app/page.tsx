"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import dynamic from "next/dynamic";
import ImageUpload from "./components/ImageUpload";
import BeforeAfterSlider from "./components/BeforeAfterSlider";
import TexturePicker, { TextureInfo } from "./components/TexturePicker";
import GeneratingOverlay from "./components/GeneratingOverlay";

const ErrorBoundary = dynamic(() => import("./components/ErrorBoundary"), { ssr: false });
const RectangleDrawer = dynamic(() => import("./components/RectangleDrawer"), {
  ssr: false,
  loading: () => <div className="h-64 sm:h-72 rounded-2xl animate-shimmer" />,
});

interface Point { x: number; y: number }
type AppStage = "upload" | "edit" | "generating" | "result";

const DEMO_ROOMS = [
  { id: "living", label: "Salon", url: "/demo-salon.png" },
  { id: "kitchen", label: "Kuchnia", url: "/demo-kuchnia.png" },
  { id: "exterior", label: "Ściana zewnętrzna", url: "/demo-exterior.png" },
];

function IconRefresh() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 102.13-9.36L1 10" /></svg>; }
function IconSparkle() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2l2.09 6.26L20 10l-5.91 1.74L12 18l-2.09-6.26L4 10l5.91-1.74L12 2z" /></svg>; }

const STEPS = [
  { n: 1, label: "Zdjęcie" },
  { n: 2, label: "Zaznacz & tekstura" },
  { n: 3, label: "Wizualizacja" },
];

function StepBar({ step }: { step: number }) {
  return (
    <div className="bg-white border-b border-stone-200/60">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-3 sm:py-4">
        <div className="flex items-center">
          {STEPS.map(({ n, label }, i) => (
            <div key={n} className="flex items-center flex-1 last:flex-none">
              <div className="flex flex-col items-center gap-1.5 shrink-0">
                <span className={`w-7 h-7 sm:w-8 sm:h-8 rounded-full flex items-center justify-center text-[11px] sm:text-xs font-bold transition-all ${n < step ? "bg-[#A01B1B] text-white" : n === step ? "border-2 border-[#A01B1B] text-[#A01B1B] bg-white" : "border border-stone-200 text-stone-400 bg-stone-50"}`}>
                  {n < step ? <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg> : n}
                </span>
                <span className={`text-[10px] sm:text-[11px] font-medium whitespace-nowrap ${n < step ? "text-[#A01B1B]" : n === step ? "text-stone-800" : "text-stone-400"}`}>{label}</span>
              </div>
              {i < STEPS.length - 1 && <div className={`flex-1 h-[2px] mx-2 sm:mx-3 rounded-full transition-colors -mt-5 ${n < step ? "bg-[#A01B1B]" : "bg-stone-200"}`} />}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── EXIF helpers ── */
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

/* ── Mask generation: white rectangle on black background ── */
function generateMask(pts: Point[], stageW: number, stageH: number, imgW: number, imgH: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = imgW;
  canvas.height = imgH;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, imgW, imgH);
  const scaleX = imgW / stageW;
  const scaleY = imgH / stageH;
  const [a, b] = pts;
  const x0 = Math.min(a.x, b.x) * scaleX;
  const y0 = Math.min(a.y, b.y) * scaleY;
  const x1 = Math.max(a.x, b.x) * scaleX;
  const y1 = Math.max(a.y, b.y) * scaleY;
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  return canvas.toDataURL("image/png");
}

/* ── Load image as HTMLImageElement ── */
function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

/* ── Pre-tile texture at correct real-world scale ── */
const ASSUMED_WALL_HEIGHT_MM = 2500; // standard interior ceiling

async function createTiledTexture(
  albedoUrl: string,
  wallWidthPx: number,
  wallHeightPx: number,
  tex: TextureInfo,
): Promise<string> {
  const img = await loadImage(albedoUrl);
  const albedoW = img.naturalWidth;
  const albedoH = img.naturalHeight;
  const albedoAspect = albedoW / albedoH;

  // Determine real-world dimensions of one albedo tile
  let tileRealW: number;
  let tileRealH: number;

  if (tex.tileRealHeightMm && tex.tileRealWidthMm) {
    // Best case: exact tile dimensions specified
    tileRealW = tex.tileRealWidthMm;
    tileRealH = tex.tileRealHeightMm;
  } else if (tex.tileWidthMm && tex.tileHeightMm) {
    tileRealW = tex.tileWidthMm;
    tileRealH = tex.tileHeightMm;
  } else if (tex.moduleWidthMm && tex.moduleHeightMm) {
    // Compute from module dimensions + number of courses in the albedo
    const courses = tex.albedoBrickCourses || 4;
    const joint = tex.jointMm || 1;
    tileRealH = courses * (tex.moduleHeightMm + joint);
    tileRealW = tileRealH * albedoAspect;
  } else {
    // Fallback: assume tile covers ~500mm
    tileRealH = 500;
    tileRealW = 500 * albedoAspect;
  }

  // Apply scale multiplier from metadata
  const scaleMult = tex.textureScaleMultiplier || 1.0;
  tileRealW *= scaleMult;
  tileRealH *= scaleMult;

  // Convert real mm to pixels on the wall canvas
  // mmPerPixel = how many mm each pixel of the wall represents
  const mmPerPixel = ASSUMED_WALL_HEIGHT_MM / wallHeightPx;
  const tilePxW = tileRealW / mmPerPixel;
  const tilePxH = tileRealH / mmPerPixel;

  // Create canvas and tile
  const canvas = document.createElement("canvas");
  canvas.width = wallWidthPx;
  canvas.height = wallHeightPx;
  const ctx = canvas.getContext("2d")!;

  const cols = Math.ceil(wallWidthPx / tilePxW) + 1;
  const rows = Math.ceil(wallHeightPx / tilePxH) + 1;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      ctx.drawImage(img, col * tilePxW, row * tilePxH, tilePxW, tilePxH);
    }
  }

  return canvas.toDataURL("image/jpeg", 0.92);
}

/* ══════════════════════════════════════════════════════════════════════════════ */

export default function Home() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [originalImageBase64, setOriginalImageBase64] = useState<string | null>(null);
  const [rectPts, setRectPts] = useState<Point[]>([]);
  const [stageSize, setStageSizeState] = useState<{ width: number; height: number } | null>(null);
  const [stage, setStage] = useState<AppStage>("upload");
  const [error, setError] = useState<string | null>(null);
  const [demoLoading, setDemoLoading] = useState<string | null>(null);
  const [selectedTexture, setSelectedTexture] = useState<TextureInfo | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const prevUrl = useRef<string | null>(null);

  const handleImageSelected = useCallback((objectUrl: string) => {
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    prevUrl.current = objectUrl;
    setImageSrc(objectUrl);
    setRectPts([]);
    setStage("edit");
    setError(null);
    setResultImage(null);
  }, []);

  useEffect(() => { return () => { if (prevUrl.current) URL.revokeObjectURL(prevUrl.current); }; }, []);

  const handleDemoRoom = useCallback(async (url: string, id: string) => {
    setDemoLoading(id);
    try {
      const resp = await fetch(url);
      const blob = await resp.blob();
      const objectUrl = URL.createObjectURL(blob);
      handleImageSelected(objectUrl);
    } catch { /* fail silently */ } finally { setDemoLoading(null); }
  }, [handleImageSelected]);

  function resetAll() {
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    prevUrl.current = null;
    setImageSrc(null);
    setOriginalImageBase64(null);
    setRectPts([]);
    setStage("upload");
    setError(null);
    setSelectedTexture(null);
    setResultImage(null);
  }

  const handleStageSizeChange = useCallback((size: { width: number; height: number }) => setStageSizeState(size), []);
  const wallDefined = rectPts.length >= 2;

  const handleGenerate = useCallback(async () => {
    if (!imageSrc || !wallDefined || !stageSize || !selectedTexture) return;
    setError(null);
    setStage("generating");

    try {
      // Prepare room image
      const roomBase64 = await prepareImageForUpload(imageSrc);
      setOriginalImageBase64(roomBase64);

      // Get actual image dimensions from the base64
      const roomImgEl = await loadImage(roomBase64);
      const imgW = roomImgEl.naturalWidth;
      const imgH = roomImgEl.naturalHeight;

      // Wall rect in actual image pixels
      const [a, b] = rectPts;
      const scaleX = imgW / stageSize.width;
      const scaleY = imgH / stageSize.height;
      const wallX = Math.min(a.x, b.x) * scaleX;
      const wallY = Math.min(a.y, b.y) * scaleY;
      const wallW = Math.abs(b.x - a.x) * scaleX;
      const wallH = Math.abs(b.y - a.y) * scaleY;

      // Generate mask
      const maskBase64 = generateMask(rectPts, stageSize.width, stageSize.height, imgW, imgH);

      // Pre-tile texture at correct scale
      const tiledTextureBase64 = await createTiledTexture(
        selectedTexture.albedoUrl,
        Math.round(wallW),
        Math.round(wallH),
        selectedTexture,
      );

      // ── Client-side composite: paste tiled texture onto room within exact mask ──
      const tiledImg = await loadImage(tiledTextureBase64);
      const compositeCanvas = document.createElement("canvas");
      compositeCanvas.width = imgW;
      compositeCanvas.height = imgH;
      const cctx = compositeCanvas.getContext("2d")!;

      // 1. Draw original room
      cctx.drawImage(roomImgEl, 0, 0);

      // 2. Clip to exact wall rectangle and draw tiled texture
      cctx.save();
      cctx.beginPath();
      cctx.rect(Math.round(wallX), Math.round(wallY), Math.round(wallW), Math.round(wallH));
      cctx.clip();
      cctx.drawImage(tiledImg, Math.round(wallX), Math.round(wallY), Math.round(wallW), Math.round(wallH));
      cctx.restore();

      const compositeBase64 = compositeCanvas.toDataURL("image/jpeg", 0.92);

      // Call API with pre-composited image
      const resp = await fetch("/api/visualize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          roomImageBase64: roomBase64,
          compositeImageBase64: compositeBase64,
          maskImageBase64: maskBase64,
          textureName: selectedTexture.name,
          textureDescription: selectedTexture.surfaceDescription || "",
          materialType: selectedTexture.materialType || "",
          wallWidthPx: Math.round(wallW),
          wallHeightPx: Math.round(wallH),
          imageWidthPx: imgW,
          imageHeightPx: imgH,
        }),
      });

      const data = await resp.json();
      if (!resp.ok || data.error) {
        throw new Error(data.error || "Błąd API");
      }

      setResultImage(data.image);
      setStage("result");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generowanie nie powiodło się");
      setStage("edit");
    }
  }, [imageSrc, wallDefined, stageSize, rectPts, selectedTexture]);

  const step = stage === "upload" ? 1 : stage === "edit" ? 2 : 3;

  return (
    <div className="h-[100svh] flex flex-col bg-[#faf9f7] overflow-hidden">
      {/* Header */}
      <header className="stegu-gradient shrink-0 z-40">
        <div className="max-w-[1440px] mx-auto px-3 sm:px-6 lg:px-8 h-12 sm:h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <img src="/stegu-logo.png" alt="Stegu" className="h-6 sm:h-7" />
            <div className="w-px h-5 bg-stone-300" />
            <span className="text-stone-500 text-[10px] sm:text-[11px] font-medium tracking-widest uppercase">Visualizer</span>
          </div>
          <div className="flex items-center gap-3">
            {stage !== "upload" && (
              <button type="button" onClick={resetAll} className="text-[11px] text-stone-500 hover:text-stone-800 transition-colors cursor-pointer">Zacznij od nowa</button>
            )}
          </div>
        </div>
      </header>

      <StepBar step={step} />

      {/* Main */}
      <main className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-[1440px] w-full mx-auto px-3 sm:px-6 lg:px-8 py-3 sm:py-5">

          {/* UPLOAD */}
          {stage === "upload" && (
            <div className="animate-fade-in-up max-w-2xl mx-auto flex flex-col">
              <div className="text-center mb-3 sm:mb-5">
                <h1 className="text-2xl sm:text-3xl font-semibold text-stone-800 tracking-tight font-[family-name:var(--font-outfit)]">
                  Zobacz produkt na swojej ścianie
                </h1>
                <p className="text-stone-500 mt-1.5 text-xs sm:text-sm max-w-lg mx-auto leading-relaxed hidden sm:block">
                  Wgraj swoje zdjęcie, zaznacz ścianę i sprawdź jak będzie wyglądał wybrany produkt Stegu — w realistycznej wizualizacji AI.
                </p>
                <div className="flex items-center justify-center gap-4 mt-2 text-[10px] text-stone-400">
                  <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#A01B1B]/40" /> Bezpłatne</span>
                  <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#A01B1B]/40" /> Bez rejestracji</span>
                  <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-[#A01B1B]/40" /> Wynik w ~20s</span>
                </div>
              </div>
              <ImageUpload onImageSelected={handleImageSelected} hasImage={false} />
              <div className="mt-3 sm:mt-5">
                <div className="flex items-center gap-2 mb-2 sm:mb-3">
                  <div className="h-px flex-1 bg-stone-200" />
                  <span className="text-[10px] text-stone-400 font-medium px-2 whitespace-nowrap">lub wypróbuj przykładowe zdjęcie</span>
                  <div className="h-px flex-1 bg-stone-200" />
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {DEMO_ROOMS.map(room => (
                    <button key={room.id} type="button" onClick={() => handleDemoRoom(room.url, room.id)} disabled={demoLoading !== null}
                      className="group relative rounded-xl overflow-hidden cursor-pointer aspect-[4/3] border border-stone-200 hover:border-stone-300 transition-all hover:shadow-md disabled:opacity-60">
                      <img src={room.url} alt={room.label} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" loading="lazy" />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                      <span className="absolute bottom-1.5 left-2 text-[10px] font-semibold text-white drop-shadow-sm">
                        {demoLoading === room.id ? "Ładowanie…" : room.label}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="mt-4 hidden sm:grid grid-cols-3 gap-2 sm:gap-3">
                {[
                  { title: "Wgraj zdjęcie", desc: "Własne lub z przykładów", num: "1" },
                  { title: "Zaznacz ścianę", desc: "Wybierz produkt Stegu", num: "2" },
                  { title: "Wizualizacja AI", desc: "Gemini generuje wynik", num: "3" },
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

          {/* EDIT */}
          {stage === "edit" && (
            <div className="animate-fade-in max-w-4xl mx-auto flex flex-col gap-4">
              <div className="bg-white rounded-2xl border border-stone-200/80 shadow-sm overflow-hidden flex flex-col">
                <div className="px-4 sm:px-5 py-3 border-b border-stone-100 flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <h2 className="text-sm font-semibold text-stone-700">Zaznacz ścianę</h2>
                    {wallDefined && (
                      <span className="px-2 py-0.5 rounded-full bg-[#A01B1B]/10 text-[#A01B1B] text-[10px] font-semibold">Ściana zaznaczona</span>
                    )}
                  </div>
                  <button type="button" onClick={() => { setImageSrc(null); setStage("upload"); }} className="text-[11px] text-stone-400 hover:text-stone-600 cursor-pointer transition-colors">Zmień zdjęcie</button>
                </div>
                <div className="relative flex-1 min-h-0 overflow-y-auto p-2 sm:p-3" style={{ minHeight: "200px" }}>
                  {imageSrc && (
                    <ErrorBoundary>
                      <p className="text-[11px] text-stone-400 mb-2.5">Narysuj prostokąt na ścianie — przeciągaj narożniki aby dopasować</p>
                      <RectangleDrawer imageSrc={imageSrc} points={rectPts} onPointsChange={setRectPts} onStageSizeChange={handleStageSizeChange} />
                    </ErrorBoundary>
                  )}
                </div>
              </div>

              {/* Texture picker */}
              <div className="bg-white rounded-2xl border border-stone-200/80 shadow-sm overflow-hidden p-3 sm:p-4">
                <TexturePicker selected={selectedTexture} onSelect={setSelectedTexture} />
              </div>

              {/* Generate button + error */}
              <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                {wallDefined && selectedTexture ? (
                  <button type="button" onClick={handleGenerate}
                    className="px-5 sm:px-7 py-2.5 text-xs sm:text-sm font-semibold text-white bg-[#A01B1B] rounded-xl hover:bg-[#8A1717] shadow-sm transition-all hover:shadow-md cursor-pointer flex items-center gap-2 flex-1 sm:flex-none justify-center">
                    <IconSparkle /> Generuj wizualizację
                  </button>
                ) : (
                  <p className="text-[11px] text-stone-400">
                    {!wallDefined ? "Narysuj prostokąt na zdjęciu aby zaznaczyć ścianę" : "Wybierz teksturę z listy powyżej"}
                  </p>
                )}
                {selectedTexture && (
                  <div className="flex items-center gap-2 ml-auto">
                    <img src={selectedTexture.albedoUrl} alt="" className="w-8 h-8 rounded-lg border border-stone-200 object-cover" />
                    <span className="text-[11px] font-medium text-stone-600">{selectedTexture.name}</span>
                  </div>
                )}
              </div>

              {error && (
                <div className="flex items-center gap-3 w-full bg-red-50 border border-red-200 rounded-xl px-4 py-3 animate-fade-in">
                  <span className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
                  <p className="text-xs text-red-700 flex-1">{error}</p>
                </div>
              )}
            </div>
          )}

          {/* GENERATING */}
          {stage === "generating" && (
            <div className="animate-fade-in max-w-4xl mx-auto">
              <div className="bg-white rounded-2xl border border-stone-200/80 shadow-sm overflow-hidden">
                <div className="px-4 sm:px-5 py-3 border-b border-stone-100">
                  <h2 className="text-sm font-semibold text-stone-700">Generowanie wizualizacji…</h2>
                </div>
                <GeneratingOverlay />
              </div>
            </div>
          )}

          {/* RESULT */}
          {stage === "result" && resultImage && (
            <div className="animate-fade-in max-w-4xl mx-auto">
              <div className="bg-white rounded-2xl border border-stone-200/80 shadow-sm overflow-hidden flex flex-col">
                <div className="px-4 sm:px-5 py-3 border-b border-stone-100 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-stone-700">Twoja wizualizacja</h2>
                  {selectedTexture && (
                    <div className="flex items-center gap-2">
                      <img src={selectedTexture.albedoUrl} alt="" className="w-6 h-6 rounded-md border border-stone-200 object-cover" />
                      <span className="text-[11px] font-medium text-stone-500">{selectedTexture.name}</span>
                    </div>
                  )}
                </div>

                <div className="p-3 sm:p-5">
                  {originalImageBase64 && (
                    <BeforeAfterSlider before={originalImageBase64} after={resultImage} />
                  )}
                </div>

                <div className="px-3 sm:px-5 py-3 border-t border-stone-100 flex flex-wrap items-center gap-2 sm:gap-3 bg-stone-50/50">
                  <button type="button" onClick={() => { setRectPts([]); setResultImage(null); setStage("edit"); setError(null); }}
                    className="px-4 py-2 text-xs font-medium text-stone-500 border border-stone-200 rounded-xl hover:bg-stone-50 transition-all cursor-pointer flex items-center gap-2">
                    <IconRefresh /> Zmień zaznaczenie
                  </button>
                  <button type="button" onClick={resetAll} className="px-3 py-2 text-xs text-stone-400 hover:text-stone-600 cursor-pointer transition-colors">
                    Nowe zdjęcie
                  </button>
                  {/* Download */}
                  <a href={resultImage} download="stegu-wizualizacja.png"
                    className="ml-auto px-4 py-2 text-xs font-semibold text-white bg-[#A01B1B] rounded-xl hover:bg-[#8A1717] shadow-sm transition-all cursor-pointer flex items-center gap-2">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Pobierz
                  </a>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-stone-200/60 py-2 sm:py-2.5 bg-white shrink-0">
        <div className="max-w-[1440px] mx-auto px-3 sm:px-6 lg:px-8 flex items-center justify-between">
          <p className="text-[9px] text-stone-400 tracking-wide">Stegu Visualizer · Kamień dekoracyjny i cegła · Powered by Gemini AI</p>
        </div>
      </footer>
    </div>
  );
}
