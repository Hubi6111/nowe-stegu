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
}
type Stage = "upload" | "edit" | "rendering" | "generated";
interface RenderData {
  composite: string | null;
  refined: string | null;
  geminiModel: string;
  timings: Record<string, number>;
}
type ResultTab = "final" | "composite" | "compare";

async function blobUrlToBase64(url: string): Promise<string> {
  const res = await fetch(url);
  const blob = await res.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

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

function LoadingOverlay({ progress }: { progress: string }) {
  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-white/80 backdrop-blur-sm rounded-2xl animate-fade-in">
      <div className="relative w-16 h-16 mb-5">
        <div className="absolute inset-0 rounded-full border-4 border-stone-200" />
        <div className="absolute inset-0 rounded-full border-4 border-[#A01B1B] border-t-transparent animate-spin-slow" />
        <div className="absolute inset-2 rounded-full border-2 border-[#A01B1B]/30 border-b-transparent animate-spin" style={{ animationDirection: "reverse" }} />
      </div>
      <p className="text-sm font-medium text-stone-700">Generowanie wizualizacji</p>
      {progress && <p className="text-xs text-stone-400 mt-1.5">{progress}</p>}
      <div className="mt-4 flex gap-1">
        {[0, 1, 2].map(i => (
          <div key={i} className="w-1.5 h-1.5 rounded-full bg-[#A01B1B] animate-pulse" style={{ animationDelay: `${i * 200}ms` }} />
        ))}
      </div>
    </div>
  );
}

export default function Home() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
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
  const [resultTab, setResultTab] = useState<ResultTab>("final");
  const [mobileProductOpen, setMobileProductOpen] = useState(false);
  const [remaining, setRemaining] = useState<{ remaining: number; limit: number; unlimited: boolean } | null>(null);

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
    setRectPts([]);
    setStage("edit");
    setRenderData(null);
    setPipelineError(null);
    setGenProgress("");
  }, []);

  useEffect(() => { return () => { if (prevUrl.current) URL.revokeObjectURL(prevUrl.current); }; }, []);

  function resetAll() {
    if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
    prevUrl.current = null;
    setImageSrc(null);
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
      try {
        const res = await fetch(url, opts);
        if (res.ok || res.status < 500) return res;
        if (i < maxAttempts) { setGenProgress(`Ponawianie (${i}/${maxAttempts})…`); await new Promise(r => setTimeout(r, 2000 * i)); continue; }
        return res;
      } catch (e) {
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
    setGenProgress("Przygotowywanie obrazu…");

    try {
      const imageBase64 = await blobUrlToBase64(imageSrc);
      setGenProgress("Nakładanie tekstury + AI render…");

      const polygon = rectToPolygon(rectPts);

      const res = await fetchWithRetry(`${API_BASE}/api/render-final`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageBase64,
          polygon,
          product_id: selectedProduct.productId,
          canvas_width: stageSize.width,
          canvas_height: stageSize.height,
        }),
      }, 3);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(formatFastApiError(err) || `Server error ${res.status}`);
      }

      const data = await res.json();
      setRenderData({
        composite: data.composite || null,
        refined: data.refined || data.composite || null,
        geminiModel: data.gemini_model || "unknown",
        timings: data.timings || {},
      });
      setResultTab("final");
      setStage("generated");
      fetchRemaining();
    } catch (e) {
      setPipelineError(e instanceof Error ? e.message : "Generowanie nie powiodło się");
      setStage("edit");
    } finally {
      setGenProgress("");
    }
  }, [imageSrc, canGenerate, selectedProduct, stageSize, rectPts, fetchWithRetry]);

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

  const ProductGrid = () => (
    <>
      {productsLoading ? (
        <div className="grid grid-cols-3 sm:grid-cols-2 gap-2">
          {[0,1,2,3].map(i => <div key={i} className="aspect-square rounded-xl animate-shimmer" />)}
        </div>
      ) : products.length === 0 ? (
        <p className="text-xs text-stone-400 text-center py-6">Brak produktów</p>
      ) : (
        <div className="grid grid-cols-3 sm:grid-cols-2 gap-2">
          {products.map(p => {
            const active = selectedProduct?.productId === p.productId;
            return (
              <button
                key={p.productId}
                type="button"
                onClick={() => { setSelectedProduct(p); setMobileProductOpen(false); }}
                className={`group relative rounded-xl overflow-hidden transition-all cursor-pointer ${
                  active
                    ? "ring-2 ring-[#A01B1B] ring-offset-2 ring-offset-white shadow-md scale-[1.02]"
                    : "ring-1 ring-stone-200 hover:ring-stone-300 hover:shadow-sm"
                }`}
              >
                <img src={`/api/textures/${p.productId}`} alt={p.name} className="aspect-square w-full object-cover" />
                <div className={`absolute inset-x-0 bottom-0 px-1.5 py-1 ${
                  active ? "bg-[#7A1515]/90" : "bg-gradient-to-t from-black/70 via-black/30 to-transparent"
                }`}>
                  <p className="text-[9px] sm:text-[10px] font-medium text-white leading-tight truncate">{p.name}</p>
                </div>
                {active && (
                  <div className="absolute top-1 right-1 w-4 h-4 sm:w-5 sm:h-5 rounded-full bg-[#A01B1B] flex items-center justify-center">
                    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}
    </>
  );

  return (
    <div className="min-h-screen flex flex-col">
      <header className="stegu-gradient sticky top-0 z-40 shadow-lg">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 h-14 sm:h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <img src="/stegu-logo.png" alt="Stegu" className="h-6 sm:h-7" />
            <span className="text-stone-400 text-[9px] sm:text-[10px] font-medium tracking-wider ml-0.5">Visualizer</span>
          </div>
          <div className="flex items-center gap-3">
            {remaining && !remaining.unlimited && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/10 backdrop-blur-sm">
                <span className={`text-[10px] sm:text-[11px] font-semibold ${remaining.remaining <= 3 ? "text-red-400" : "text-stone-300"}`}>
                  {remaining.remaining}/{remaining.limit}
                </span>
                <span className="text-[9px] text-stone-500 hidden sm:inline">generacji</span>
              </div>
            )}
            {stage !== "upload" && (
              <button type="button" onClick={resetAll} className="text-[11px] text-stone-400 hover:text-white transition-colors cursor-pointer">
                Od początku
              </button>
            )}
            <a href="/admin" className="text-[10px] text-stone-500 hover:text-stone-300 transition-colors">
              Admin
            </a>
          </div>
        </div>
      </header>

      <div className="bg-white border-b border-stone-200/60">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 py-2.5 sm:py-3">
          <div className="flex items-center gap-1.5 sm:gap-4">
            {[
              { n: 1, label: "Zdjęcie" },
              { n: 2, label: "Zaznaczenie" },
              { n: 3, label: "Generowanie" },
              { n: 4, label: "Wynik" },
            ].map(({ n, label }, i) => (
              <div key={n} className="flex items-center gap-1 sm:gap-3">
                {i > 0 && <div className={`w-4 sm:w-10 h-px ${n <= step ? "bg-[#A01B1B]" : "bg-stone-200"}`} />}
                <div className="flex items-center gap-1">
                  <span className={`w-5 h-5 sm:w-6 sm:h-6 rounded-full flex items-center justify-center text-[9px] sm:text-[10px] font-bold transition-all ${
                    n < step ? "bg-[#A01B1B] text-white" : n === step ? "bg-[#A01B1B]/10 text-[#A01B1B] ring-2 ring-[#A01B1B]" : "bg-stone-100 text-stone-400"
                  }`}>
                    {n < step ? "✓" : n}
                  </span>
                  <span className={`text-[10px] sm:text-[11px] font-medium hidden sm:inline ${n <= step ? "text-stone-700" : "text-stone-400"}`}>
                    {label}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 py-4 sm:py-6">
        {stage === "upload" && (
          <div className="animate-fade-in-up">
            <div className="text-center mb-6 sm:mb-8">
              <h1 className="text-xl sm:text-3xl font-bold text-stone-800" style={{ fontFamily: "var(--font-playfair)" }}>
                Zwizualizuj swoje wnętrze
              </h1>
              <p className="text-stone-500 mt-2 text-xs sm:text-sm max-w-lg mx-auto">
                Wgraj zdjęcie pokoju, wybierz materiał i zobacz jak będzie wyglądał na Twojej ścianie
              </p>
            </div>
            <div className="max-w-2xl mx-auto">
              <ImageUpload onImageSelected={handleImageSelected} hasImage={false} />
            </div>
          </div>
        )}

        {stage !== "upload" && (
          <div className="flex flex-col lg:flex-row gap-4 sm:gap-6 animate-fade-in">
            <div className="flex-1 min-w-0">
              <div className="bg-white rounded-2xl border border-stone-200/80 shadow-sm overflow-hidden">
                <div className="px-4 sm:px-5 py-3 border-b border-stone-100 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <h2 className="text-xs sm:text-sm font-semibold text-stone-700">
                      {stage === "edit" ? "Zaznacz ścianę" : stage === "rendering" ? "Generowanie…" : "Wynik"}
                    </h2>
                    {stage === "edit" && wallDefined && (
                      <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 text-[9px] sm:text-[10px] font-semibold">
                        Zaznaczono
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {stage === "edit" && (
                      <button type="button" onClick={() => { setImageSrc(null); setStage("upload"); }} className="text-[10px] sm:text-xs text-stone-400 hover:text-stone-600 cursor-pointer transition-colors">
                        Zmień zdjęcie
                      </button>
                    )}
                    {stage === "edit" && (
                      <button
                        type="button"
                        onClick={() => setMobileProductOpen(true)}
                        className="lg:hidden flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-semibold text-[#A01B1B] bg-red-50 rounded-lg border border-red-200 cursor-pointer"
                      >
                        {selectedProduct ? selectedProduct.name : "Wybierz materiał"}
                        <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2"><path d="M2 4l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" /></svg>
                      </button>
                    )}
                  </div>
                </div>

                <div className="relative p-2 sm:p-4 min-h-[240px] sm:min-h-[320px]">
                  {stage === "edit" && imageSrc && (
                    <ErrorBoundary>
                      <p className="text-[10px] sm:text-[11px] text-stone-400 mb-2 sm:mb-3">
                        Kliknij i przeciągnij aby zaznaczyć prostokąt · Przeciągaj narożniki aby dopasować
                      </p>
                      <RectangleDrawer
                        imageSrc={imageSrc}
                        points={rectPts}
                        onPointsChange={setRectPts}
                        onStageSizeChange={handleStageSizeChange}
                      />
                    </ErrorBoundary>
                  )}

                  {stage === "rendering" && <LoadingOverlay progress={genProgress} />}

                  {stage === "generated" && renderData && (
                    <div className="flex flex-col gap-3 sm:gap-4 animate-scale-in">
                      <div className="border-b border-stone-200 flex overflow-x-auto">
                        {([["final", "Wynik AI"], ["composite", "Bez AI"], ["compare", "Porównanie"]] as [ResultTab, string][]).map(([tab, label]) => (
                          <button
                            key={tab}
                            type="button"
                            onClick={() => setResultTab(tab)}
                            className={`px-3 sm:px-4 py-2 text-[10px] sm:text-xs font-semibold border-b-2 -mb-px transition-colors cursor-pointer whitespace-nowrap ${
                              resultTab === tab ? "border-[#A01B1B] text-stone-800" : "border-transparent text-stone-400 hover:text-stone-600"
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>

                      {resultTab === "compare" && imageSrc && renderData.refined ? (
                        <BeforeAfterSlider before={imageSrc} after={renderData.refined} />
                      ) : resultTab === "final" && renderData.refined ? (
                        <img src={renderData.refined} alt="Wynik" className="rounded-xl w-full" />
                      ) : resultTab === "composite" && renderData.composite ? (
                        <img src={renderData.composite} alt="Kompozyt" className="rounded-xl w-full" />
                      ) : (
                        <div className="p-8 rounded-xl bg-stone-50 text-center">
                          <p className="text-sm text-stone-400">Brak danych</p>
                        </div>
                      )}

                      <p className="text-[9px] sm:text-[10px] text-stone-300">
                        Model: {renderData.geminiModel}
                        {renderData.timings.total && ` · ${renderData.timings.total}s`}
                      </p>
                    </div>
                  )}
                </div>

                <div className="px-3 sm:px-5 py-3 border-t border-stone-100 flex flex-wrap items-center gap-2 sm:gap-3 bg-stone-50/50">
                  {stage === "edit" && canGenerate && (
                    <button
                      type="button"
                      onClick={handleGenerate}
                      className="px-5 sm:px-6 py-2.5 text-xs sm:text-sm font-semibold text-white bg-[#A01B1B] rounded-xl hover:bg-[#8A1717] shadow-sm transition-all hover:shadow-md cursor-pointer flex-1 sm:flex-none"
                    >
                      Generuj wizualizację
                    </button>
                  )}
                  {stage === "edit" && wallDefined && !selectedProduct && (
                    <p className="text-[10px] sm:text-xs text-stone-400">
                      <span className="hidden lg:inline">Wybierz materiał z panelu po prawej →</span>
                      <span className="lg:hidden">Kliknij &quot;Wybierz materiał&quot; powyżej</span>
                    </p>
                  )}

                  {stage === "generated" && (
                    <div className="flex flex-wrap items-center gap-2 w-full">
                      {renderData?.refined && (
                        <button
                          type="button"
                          onClick={() => handleDownload(renderData.refined!, "final")}
                          className="px-4 sm:px-5 py-2.5 text-xs sm:text-sm font-semibold text-white bg-stone-800 rounded-xl hover:bg-stone-700 shadow-sm transition-all cursor-pointer flex items-center gap-2"
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                          Pobierz
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={handleUseResultAsInput}
                        className="px-4 sm:px-5 py-2.5 text-xs sm:text-sm font-medium text-[#A01B1B] border border-red-300 rounded-xl hover:bg-red-50 transition-all cursor-pointer flex items-center gap-2"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 102.13-9.36L1 10"/></svg>
                        <span className="hidden sm:inline">Dodaj kolejny materiał</span>
                        <span className="sm:hidden">Kolejny</span>
                      </button>
                      <button type="button" onClick={resetToEdit} className="px-3 py-2 text-xs text-stone-500 hover:text-stone-700 cursor-pointer transition-colors">
                        ← Zmień
                      </button>
                    </div>
                  )}

                  {pipelineError && (
                    <div className="flex items-center gap-3 w-full bg-red-50 border border-red-200 rounded-xl px-3 sm:px-4 py-3 animate-fade-in">
                      <span className="w-2 h-2 rounded-full bg-red-500 shrink-0" />
                      <p className="text-[10px] sm:text-xs text-red-700 flex-1">{pipelineError}</p>
                      {canGenerate && (
                        <button
                          type="button"
                          onClick={() => { setPipelineError(null); handleGenerate(); }}
                          className="text-[10px] sm:text-xs font-semibold text-red-700 underline underline-offset-2 cursor-pointer shrink-0"
                        >
                          Ponów
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="hidden lg:block w-72 xl:w-80 shrink-0">
              <div className="bg-white rounded-2xl border border-stone-200/80 shadow-sm sticky top-24">
                <div className="px-4 py-3.5 border-b border-stone-100">
                  <h3 className="text-sm font-semibold text-stone-700">Wybierz materiał</h3>
                  {selectedProduct && (
                    <p className="text-xs text-[#A01B1B] font-medium mt-0.5">{selectedProduct.name}</p>
                  )}
                </div>
                <div className="p-3 max-h-[60vh] overflow-y-auto">
                  <ProductGrid />
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {mobileProductOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setMobileProductOpen(false)} />
          <div className="absolute bottom-0 inset-x-0 bg-white rounded-t-3xl shadow-2xl animate-fade-in-up max-h-[70vh] flex flex-col">
            <div className="flex items-center justify-between px-5 py-4 border-b border-stone-100">
              <h3 className="text-sm font-semibold text-stone-700">Wybierz materiał</h3>
              <button onClick={() => setMobileProductOpen(false)} className="w-8 h-8 rounded-full bg-stone-100 flex items-center justify-center cursor-pointer">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
              </button>
            </div>
            <div className="p-4 overflow-y-auto flex-1">
              <ProductGrid />
            </div>
          </div>
        </div>
      )}

      <footer className="border-t border-stone-200/60 py-4 sm:py-5 text-center bg-white">
        <p className="text-[9px] sm:text-[10px] text-stone-400 tracking-wide">
          Stegu Visualizer · Kamień dekoracyjny i cegła
        </p>
      </footer>
    </div>
  );
}
