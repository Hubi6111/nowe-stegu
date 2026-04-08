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
        className={`flex flex-col items-center justify-center gap-4 h-52 sm:h-64 border-2 border-dashed rounded-2xl transition-all cursor-pointer ${
          dragOver
            ? "border-[#A01B1B] bg-red-50/50 scale-[1.01]"
            : "border-stone-300 bg-white hover:border-[#A01B1B]/40 hover:bg-red-50/20"
        }`}
      >
        <div className="w-14 h-14 rounded-2xl bg-stone-100 flex items-center justify-center">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-stone-400">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-stone-600">Upuść zdjęcie pokoju lub kliknij aby wybrać</p>
          <p className="text-xs text-stone-400 mt-1">JPG, PNG, WebP · max 20 MB</p>
        </div>
      </div>
      {error && <p className="text-xs text-red-600 px-1">{error}</p>}
      <input ref={inputRef} type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
    </div>
  );
}
