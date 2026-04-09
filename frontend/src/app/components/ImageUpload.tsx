"use client";

import { useRef, useCallback, useState } from "react";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

interface ImageUploadProps {
  onImageSelected: (objectUrl: string) => void;
  hasImage: boolean;
}

export default function ImageUpload({ onImageSelected, hasImage }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const processFile = useCallback((file: File | undefined | null) => {
    setError(null);
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) { setError("Nieobsługiwany format. Użyj JPG, PNG lub WebP."); return; }
    if (file.size > MAX_FILE_SIZE) { setError(`Plik za duży (${(file.size / 1024 / 1024).toFixed(1)} MB). Max 20 MB.`); return; }
    const url = URL.createObjectURL(file);
    const img = new window.Image();
    img.onload = () => onImageSelected(url);
    img.onerror = () => { URL.revokeObjectURL(url); setError("Nie udało się wczytać obrazu."); };
    img.src = url;
  }, [onImageSelected]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    processFile(e.target.files?.[0]);
    if (inputRef.current) inputRef.current.value = "";
  }, [processFile]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    processFile(e.dataTransfer.files?.[0]);
  }, [processFile]);

  if (hasImage) {
    return (
      <div className="flex items-center gap-3">
        <button type="button" onClick={() => inputRef.current?.click()} className="px-3 py-1.5 text-xs font-medium text-stone-500 border border-stone-300 rounded-lg hover:border-[#A01B1B]/50 hover:text-[#A01B1B] transition-colors cursor-pointer">
          Zmień zdjęcie
        </button>
        {error && <p className="text-xs text-red-600">{error}</p>}
        <input ref={inputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onClick={() => inputRef.current?.click()}
        className={`group flex flex-col items-center justify-center gap-3 h-32 sm:h-48 border-2 border-dashed rounded-2xl transition-all cursor-pointer ${
          dragOver
            ? "border-[#A01B1B] bg-red-50/50 scale-[1.01]"
            : "border-stone-300 bg-white hover:border-[#A01B1B]/40 hover:bg-red-50/20"
        }`}
      >
        <div className="w-10 h-10 sm:w-14 sm:h-14 rounded-xl sm:rounded-2xl bg-[#A01B1B]/8 flex items-center justify-center group-hover:bg-[#A01B1B]/12 transition-colors">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#A01B1B" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="opacity-70 sm:hidden">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
            <path d="M17 8l-5-5-5 5" />
            <path d="M12 3v12" />
          </svg>
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#A01B1B" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="opacity-70 hidden sm:block">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
            <path d="M17 8l-5-5-5 5" />
            <path d="M12 3v12" />
          </svg>
        </div>
        <div className="text-center px-4">
          <p className="text-sm font-semibold text-stone-700">
            <span className="sm:hidden">Wgraj lub zrób zdjęcie pokoju</span>
            <span className="hidden sm:inline">Upuść zdjęcie pokoju lub kliknij aby wybrać</span>
          </p>
          <p className="text-[11px] sm:text-xs text-stone-400 mt-0.5">JPG, PNG, WebP · max 20 MB</p>
        </div>
      </div>
      {error && <p className="text-xs text-red-600 px-1">{error}</p>}
      <input ref={inputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
    </div>
  );
}
