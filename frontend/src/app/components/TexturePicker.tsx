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
  "amsterdam-1",
  "arena-1",
  "boston-3",
  "cambridge-1",
  "cambridge-8",
  "linea-comfort-hazelnut",
  "linea-comfort-oak",
  "monsanto-1",
  "monsanto-2",
];

interface TexturePickerProps {
  selected: TextureInfo | null;
  onSelect: (tex: TextureInfo) => void;
}

export default function TexturePicker({ selected, onSelect }: TexturePickerProps) {
  const [textures, setTextures] = useState<TextureInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>("all");

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

  const categories = ["all", ...Array.from(new Set(textures.map(t => t.category || "other")))];
  const filtered = filter === "all" ? textures : textures.filter(t => t.category === filter);

  const categoryLabels: Record<string, string> = {
    all: "Wszystkie",
    brick: "Cegła",
    stone: "Kamień",
    wood: "Drewno",
    other: "Inne",
  };

  const handleSelect = useCallback((tex: TextureInfo) => {
    onSelect(tex);
  }, [onSelect]);

  if (loading) {
    return (
      <div className="animate-fade-in">
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-sm font-semibold text-stone-700">Wybierz teksturę</h3>
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2">
          {Array.from({ length: 9 }).map((_, i) => (
            <div key={i} className="aspect-square rounded-xl animate-shimmer" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <h3 className="text-sm font-semibold text-stone-700">Wybierz teksturę</h3>
        <div className="flex items-center gap-1 ml-auto">
          {categories.map(cat => (
            <button
              key={cat}
              type="button"
              onClick={() => setFilter(cat)}
              className={`px-2.5 py-1 text-[10px] font-medium rounded-lg transition-all cursor-pointer ${
                filter === cat
                  ? "bg-[#A01B1B] text-white shadow-sm"
                  : "text-stone-500 hover:bg-stone-100"
              }`}
            >
              {categoryLabels[cat] || cat}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2">
        {filtered.map(tex => {
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
              {/* Name label */}
              <span className={`absolute bottom-1 left-1.5 right-1.5 text-[9px] sm:text-[10px] font-semibold text-white drop-shadow-sm truncate transition-opacity ${
                isSelected ? "opacity-100" : "opacity-0 group-hover:opacity-100"
              }`}>
                {tex.name}
              </span>
              {/* Selected checkmark */}
              {isSelected && (
                <div className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full bg-[#A01B1B] flex items-center justify-center shadow-sm">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                </div>
              )}
              {/* Category badge */}
              <span className={`absolute top-1.5 left-1.5 px-1.5 py-0.5 text-[7px] font-bold uppercase tracking-wider rounded-md backdrop-blur-sm transition-opacity ${
                isSelected
                  ? "bg-white/90 text-[#A01B1B] opacity-100"
                  : "bg-black/40 text-white/80 opacity-0 group-hover:opacity-100"
              }`}>
                {categoryLabels[tex.category || "other"] || tex.category}
              </span>
            </button>
          );
        })}
      </div>

      {filtered.length === 0 && (
        <p className="text-center text-xs text-stone-400 py-6">
          Brak tekstur w wybranej kategorii
        </p>
      )}
    </div>
  );
}
