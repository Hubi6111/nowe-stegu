"use client";

import { useState, useEffect, useCallback } from "react";

export interface TextureInfo {
  id: string;
  name: string;
  category?: string;
  materialType?: string;
  surfaceDescription?: string;
  albedoUrl: string;
  shopUrl?: string;
  moduleWidthMm?: number;
  moduleHeightMm?: number;
  tileWidthMm?: number;
  tileHeightMm?: number;
  tileRealHeightMm?: number;
  tileRealWidthMm?: number;
  layoutType?: string;
  textureScaleMultiplier?: number;
  jointMm?: number;
  albedoBrickCourses?: number;
}

const TEXTURE_IDS = [
  "boston-3",
  "cambridge-1",
  "cambridge-8",
  "linea-comfort-hazelnut",
  "linea-comfort-oak",
  "monsanto-1",
  "monsanto-2",
];

/* ── Category config ── */
const CATEGORY_ORDER = ["brick", "stone", "wood"];
const CATEGORY_LABELS: Record<string, string> = {
  brick: "Cegły",
  stone: "Kamień",
  wood: "Lamele",
  other: "Inne",
};

function CategoryIcon({ cat, className = "" }: { cat: string; className?: string }) {
  const cls = `w-4 h-4 ${className}`;
  switch (cat) {
    case "brick":
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <rect x="2" y="4" width="20" height="5" rx="1" /><rect x="2" y="11" width="9" height="5" rx="1" /><rect x="13" y="11" width="9" height="5" rx="1" /><rect x="6" y="18" width="14" height="3" rx="1" />
        </svg>
      );
    case "stone":
      return (
        <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 8c0-1 1-3 4-3s3 2 5 2 3-2 5-2 4 1 4 3v7c0 2-2 4-4 4H8c-2 0-4-2-4-4V8z" />
          <path d="M8 13h3M14 10h2" />
        </svg>
      );
    case "wood":
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
}

interface TexturePickerProps {
  selected: TextureInfo | null;
  onSelect: (tex: TextureInfo) => void;
}

export default function TexturePicker({ selected, onSelect }: TexturePickerProps) {
  const [textures, setTextures] = useState<TextureInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [openCategories, setOpenCategories] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let cancelled = false;

    async function loadTextures() {
      const results: TextureInfo[] = [];
      for (const id of TEXTURE_IDS) {
        try {
          const resp = await fetch(`/textures/${id}/metadata.json`);
          if (!resp.ok) continue;
          const meta = await resp.json();
          results.push({
            id,
            name: meta.name || id,
            category: meta.category || "other",
            materialType: meta.materialType || "",
            surfaceDescription: meta.surfaceDescription || "",
            albedoUrl: `/textures/${id}/albedo.jpg`,
            shopUrl: meta.shopUrl || undefined,
            moduleWidthMm: meta.moduleWidthMm,
            moduleHeightMm: meta.moduleHeightMm,
            tileWidthMm: meta.tileWidthMm,
            tileHeightMm: meta.tileHeightMm,
            tileRealHeightMm: meta.tileRealHeightMm,
            tileRealWidthMm: meta.tileRealWidthMm,
            layoutType: meta.layoutType,
            textureScaleMultiplier: meta.textureScaleMultiplier,
            jointMm: meta.jointMm,
            albedoBrickCourses: meta.albedoBrickCourses,
          });
        } catch {
          // Skip textures that fail to load
        }
      }
      if (!cancelled) {
        setTextures(results);
        setLoading(false);
      }
    }

    loadTextures();
    return () => { cancelled = true; };
  }, []);

  const toggleCategory = useCallback((cat: string) => {
    setOpenCategories(prev => ({ ...prev, [cat]: !prev[cat] }));
  }, []);

  // Group by category
  const grouped = textures.reduce<Record<string, TextureInfo[]>>((acc, t) => {
    const cat = t.category || "other";
    if (!acc[cat]) acc[cat] = [];
    acc[cat].push(t);
    return acc;
  }, {});

  // Initialize categories — all collapsed by default
  useEffect(() => {
    if (textures.length > 0 && Object.keys(openCategories).length === 0) {
      const initial: Record<string, boolean> = {};
      Object.keys(grouped).forEach(cat => { initial[cat] = false; });
      setOpenCategories(initial);
    }
  }, [textures.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelect = useCallback((tex: TextureInfo) => {
    onSelect(tex);
  }, [onSelect]);

  if (loading) {
    return (
      <div className="animate-fade-in">
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-sm font-semibold text-stone-700">Wybierz teksturę</h3>
        </div>
        <div className="space-y-2">
          {[0, 1, 2].map(i => (
            <div key={i} className="rounded-xl border border-stone-100 p-3">
              <div className="h-4 w-24 rounded animate-shimmer mb-2" />
              <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                {[0, 1].map(j => <div key={j} className="aspect-square rounded-lg animate-shimmer" />)}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  function pluralProducts(n: number): string {
    if (n === 1) return "produkt";
    if (n < 5) return "produkty";
    return "produktów";
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-sm font-semibold text-stone-700">Wybierz teksturę</h3>
      </div>

      <div className="space-y-2">
        {CATEGORY_ORDER.filter(cat => grouped[cat]).map(cat => {
          const items = grouped[cat];
          const isOpen = openCategories[cat] !== false;
          const hasSelected = items.some(t => selected?.id === t.id);

          return (
            <div key={cat} className={`rounded-xl border transition-colors ${
              hasSelected ? "border-[#A01B1B]/20 bg-[#A01B1B]/[0.02]" : "border-stone-100 bg-white"
            }`}>
              {/* Category header — clickable to toggle */}
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
                  }`}>{CATEGORY_LABELS[cat] || cat}</p>
                  <p className="text-[9px] text-stone-400 mt-0.5">{items.length} {pluralProducts(items.length)}</p>
                </div>
                {/* Chevron */}
                <svg
                  className={`w-3.5 h-3.5 text-stone-300 transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}
                  viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2"
                >
                  <path d="M2 4l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>

              {/* Texture grid — shown when open */}
              {isOpen && (
                <div className="px-2.5 pb-2.5">
                  <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                    {items.map(tex => {
                      const isSelected = selected?.id === tex.id;
                      return (
                        <button
                          key={tex.id}
                          type="button"
                          onClick={() => handleSelect(tex)}
                          className={`group relative aspect-square rounded-xl overflow-hidden border-2 transition-all cursor-pointer ${
                            isSelected
                              ? "border-[#A01B1B] ring-2 ring-[#A01B1B]/20 shadow-md scale-[1.02]"
                              : "border-stone-200 hover:border-stone-300 hover:shadow-sm"
                          }`}
                        >
                          <img
                            src={tex.albedoUrl}
                            alt={tex.name}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                            loading="lazy"
                          />
                          {/* Gradient overlay */}
                          <div className={`absolute inset-0 bg-gradient-to-t transition-opacity duration-200 ${
                            isSelected ? "from-[#A01B1B]/70 to-transparent" : "from-black/50 to-transparent opacity-0 group-hover:opacity-100"
                          }`} />
                          {/* Name */}
                          <span className={`absolute bottom-1 left-1.5 right-1.5 text-[9px] sm:text-[10px] font-semibold text-white drop-shadow-sm truncate transition-opacity ${
                            isSelected ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                          }`}>
                            {tex.name}
                          </span>
                          {/* Checkmark */}
                          {isSelected && (
                            <div className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-[#A01B1B] flex items-center justify-center shadow-sm">
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="20 6 9 17 4 12" />
                              </svg>
                            </div>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {textures.length === 0 && (
        <p className="text-center text-xs text-stone-400 py-6">
          Brak dostępnych tekstur
        </p>
      )}
    </div>
  );
}
