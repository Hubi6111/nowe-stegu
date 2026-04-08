"use client";

import { useState, useEffect, useCallback, useRef } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "";
const RED = "#A01B1B";

type Tab = "stats" | "textures" | "generations" | "watermark" | "limits";

interface TextureMeta {
  name?: string;
  moduleWidthMm?: number;
  moduleHeightMm?: number;
  jointMm?: number;
  layoutType?: string;
  offsetRatio?: number;
  textureScaleMultiplier?: number;
  albedoBrickCourses?: number;
  tags?: string[];
  [key: string]: unknown;
}
interface Texture { id: string; name: string; has_albedo: boolean; meta: TextureMeta }
interface WmConfig { enabled: boolean; opacity: number; has_file: boolean }
interface LimitsData { daily_limit: number; usage: { date: string; users: Record<string, number>; total: number } }
interface GenItem { id: string; timestamp: string; client_ip: string; product_id: string; product_name: string; gemini_model: string; timings: Record<string, number>; has_image: boolean }
interface StatsData {
  total: number; today: number; unique_ips: number;
  by_product: Record<string, number>; by_day: Record<string, number>;
  daily_limit: number;
  usage_today: { date: string; users: Record<string, number>; total: number };
}

function api(path: string, pwd: string, opts: RequestInit = {}) {
  return fetch(`${API}/api/admin${path}`, {
    ...opts,
    headers: { "Content-Type": "application/json", "x-admin-password": pwd, ...(opts.headers || {}) },
  });
}

export default function AdminPage() {
  const [password, setPassword] = useState("");
  const [authed, setAuthed] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [tab, setTab] = useState<Tab>("stats");

  const handleLogin = async () => {
    setLoginError("");
    try {
      const r = await fetch(`${API}/api/admin/login`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!r.ok) { setLoginError("Nieprawidłowe hasło"); return; }
      setAuthed(true);
    } catch { setLoginError("Brak połączenia z serwerem"); }
  };

  if (!authed) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#faf9f7]">
        <div className="bg-white rounded-2xl shadow-lg border border-stone-200 p-8 w-full max-w-sm animate-fade-in">
          <div className="flex items-center gap-2 mb-6">
            <img src="/stegu-logo.png" alt="Stegu" className="h-8" />
            <span className="text-stone-400 text-[10px] ml-1">Admin</span>
          </div>
          <label className="text-xs font-medium text-stone-600 mb-1 block">Hasło administratora</label>
          <input
            type="password" value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleLogin()}
            className="w-full px-4 py-2.5 rounded-xl border border-stone-300 text-sm focus:outline-none focus:ring-2 focus:border-transparent mb-3"
            style={{ "--tw-ring-color": RED } as React.CSSProperties}
            placeholder="••••••••" autoFocus
          />
          {loginError && <p className="text-xs text-red-600 mb-2">{loginError}</p>}
          <button onClick={handleLogin} className="w-full py-2.5 text-sm font-semibold text-white rounded-xl hover:opacity-90 transition cursor-pointer" style={{ backgroundColor: RED }}>
            Zaloguj się
          </button>
          <a href="/" className="block text-center text-xs text-stone-400 mt-4 hover:text-stone-600">← Powrót do wizualizera</a>
        </div>
      </div>
    );
  }

  const tabs: [Tab, string][] = [["stats", "Statystyki"], ["textures", "Tekstury"], ["generations", "Generacje"], ["watermark", "Watermark"], ["limits", "Limity"]];

  return (
    <div className="min-h-screen bg-[#faf9f7]">
      <header className="stegu-gradient shadow-lg">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img src="/stegu-logo.png" alt="Stegu" className="h-7" />
            <span className="text-stone-400 text-[10px] font-medium">Admin</span>
          </div>
          <div className="flex items-center gap-4">
            <a href="/" className="text-xs text-stone-400 hover:text-white transition-colors">← Visualizer</a>
            <button onClick={() => { setAuthed(false); setPassword(""); }} className="text-xs text-stone-500 hover:text-stone-300 cursor-pointer transition-colors">Wyloguj</button>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6">
        <div className="flex gap-1 mb-6 bg-white rounded-xl border border-stone-200 p-1 w-fit overflow-x-auto">
          {tabs.map(([t, l]) => (
            <button key={t} onClick={() => setTab(t)} className={`px-3 sm:px-4 py-2 text-[10px] sm:text-xs font-semibold rounded-lg transition-all cursor-pointer whitespace-nowrap ${
              tab === t ? "text-white shadow-sm" : "text-stone-500 hover:text-stone-700"
            }`} style={tab === t ? { backgroundColor: RED } : {}}>{l}</button>
          ))}
        </div>

        {tab === "stats" && <StatsPanel pwd={password} />}
        {tab === "textures" && <TexturesPanel pwd={password} />}
        {tab === "generations" && <GenerationsPanel pwd={password} />}
        {tab === "watermark" && <WatermarkPanel pwd={password} />}
        {tab === "limits" && <LimitsPanel pwd={password} />}
      </div>
    </div>
  );
}

// ── Stats Panel ──────────────────────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-stone-200 p-4">
      <p className="text-[10px] font-medium text-stone-400 uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold text-stone-800 mt-1">{value}</p>
      {sub && <p className="text-[10px] text-stone-400 mt-0.5">{sub}</p>}
    </div>
  );
}

function StatsPanel({ pwd }: { pwd: string }) {
  const [data, setData] = useState<StatsData | null>(null);

  useEffect(() => {
    api("/stats", pwd).then(r => r.ok ? r.json() : null).then(setData).catch(() => {});
  }, [pwd]);

  if (!data) return <div className="h-40 rounded-xl animate-shimmer" />;

  const dayEntries = Object.entries(data.by_day).slice(0, 14);
  const maxDay = Math.max(...dayEntries.map(([, v]) => v), 1);
  const productEntries = Object.entries(data.by_product).slice(0, 10);
  const maxProd = Math.max(...productEntries.map(([, v]) => v), 1);
  const todayUsers = Object.entries(data.usage_today.users || {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="animate-fade-in space-y-6">
      <h2 className="text-lg font-bold text-stone-800">Statystyki</h2>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Wszystkie generacje" value={data.total} />
        <StatCard label="Dzisiaj" value={data.today} sub={data.usage_today.date} />
        <StatCard label="Unikalni użytkownicy" value={data.unique_ips} sub="wszystkich" />
        <StatCard label="Dzienny limit" value={data.daily_limit || "∞"} sub="na użytkownika" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Daily chart */}
        <div className="bg-white rounded-xl border border-stone-200 p-4">
          <p className="text-xs font-semibold text-stone-700 mb-3">Generacje dziennie (ostatnie 14 dni)</p>
          {dayEntries.length === 0 ? <p className="text-xs text-stone-400">Brak danych</p> : (
            <div className="space-y-1.5">
              {dayEntries.map(([day, count]) => (
                <div key={day} className="flex items-center gap-2 text-[10px]">
                  <span className="w-16 text-stone-400 shrink-0 font-mono">{day.slice(5)}</span>
                  <div className="flex-1 h-4 bg-stone-100 rounded-full overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${(count / maxDay) * 100}%`, backgroundColor: RED }} />
                  </div>
                  <span className="w-6 text-right font-semibold text-stone-600">{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Product popularity */}
        <div className="bg-white rounded-xl border border-stone-200 p-4">
          <p className="text-xs font-semibold text-stone-700 mb-3">Najpopularniejsze produkty</p>
          {productEntries.length === 0 ? <p className="text-xs text-stone-400">Brak danych</p> : (
            <div className="space-y-1.5">
              {productEntries.map(([name, count]) => (
                <div key={name} className="flex items-center gap-2 text-[10px]">
                  <span className="w-28 text-stone-500 shrink-0 truncate">{name}</span>
                  <div className="flex-1 h-4 bg-stone-100 rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-stone-700" style={{ width: `${(count / maxProd) * 100}%` }} />
                  </div>
                  <span className="w-6 text-right font-semibold text-stone-600">{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Active users today */}
      <div className="bg-white rounded-xl border border-stone-200 p-4">
        <p className="text-xs font-semibold text-stone-700 mb-3">Dzisiejsi użytkownicy ({todayUsers.length})</p>
        {todayUsers.length === 0 ? <p className="text-xs text-stone-400">Brak generacji dzisiaj</p> : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {todayUsers.map(([ip, count]) => (
              <div key={ip} className="flex items-center justify-between text-xs py-2 px-3 bg-stone-50 rounded-lg">
                <span className="font-mono text-stone-500 text-[10px]">{ip}</span>
                <span className="font-semibold text-stone-700">{count} / {data.daily_limit || "∞"}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Textures Panel ────────────────────────────────────────────────────────────

const META_FIELDS: { key: string; label: string; type: "text" | "number"; step?: number }[] = [
  { key: "name", label: "Nazwa", type: "text" },
  { key: "moduleWidthMm", label: "Szer. modułu (mm)", type: "number" },
  { key: "moduleHeightMm", label: "Wys. modułu (mm)", type: "number" },
  { key: "jointMm", label: "Fuga (mm)", type: "number" },
  { key: "layoutType", label: "Układ", type: "text" },
  { key: "offsetRatio", label: "Przesunięcie (0–1)", type: "number", step: 0.1 },
  { key: "textureScaleMultiplier", label: "Mnożnik skali", type: "number", step: 0.1 },
  { key: "albedoBrickCourses", label: "Warstwy cegieł", type: "number" },
];

const ALL_TAGS = ["wewnętrzne", "zewnętrzne", "cegła", "kamień", "lamele", "panele"];

function TexturesPanel({ pwd }: { pwd: string }) {
  const [textures, setTextures] = useState<Texture[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editMeta, setEditMeta] = useState<TextureMeta>({});
  const [saving, setSaving] = useState(false);
  const [showAdd, setShowAdd] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try { const r = await api("/textures", pwd); if (r.ok) setTextures(await r.json()); } catch {}
    setLoading(false);
  }, [pwd]);

  useEffect(() => { load(); }, [load]);

  const startEdit = (t: Texture) => { setEditingId(t.id); setEditMeta({ ...t.meta }); };
  const cancelEdit = () => { setEditingId(null); setEditMeta({}); };

  const handleSave = async () => {
    if (!editingId) return;
    setSaving(true);
    await api(`/textures/${editingId}`, pwd, { method: "PUT", body: JSON.stringify(editMeta) });
    setSaving(false); setEditingId(null); load();
  };

  const handleDelete = async (id: string) => {
    if (!confirm(`Usunąć teksturę "${id}"?`)) return;
    await api(`/textures/${id}`, pwd, { method: "DELETE" }); load();
  };

  const setField = (key: string, val: string, type: string) => {
    setEditMeta(prev => ({ ...prev, [key]: type === "number" ? (val === "" ? undefined : Number(val)) : val }));
  };

  const toggleTag = (tag: string) => {
    setEditMeta(prev => {
      const current = Array.isArray(prev.tags) ? prev.tags : [];
      const next = current.includes(tag) ? current.filter(t => t !== tag) : [...current, tag];
      return { ...prev, tags: next };
    });
  };

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-4 gap-2 flex-wrap">
        <h2 className="text-lg font-bold text-stone-800">Tekstury ({textures.length})</h2>
        <button onClick={() => setShowAdd(!showAdd)} className="px-4 py-2 text-xs font-semibold text-white rounded-xl hover:opacity-90 cursor-pointer" style={{ backgroundColor: RED }}>
          + Dodaj
        </button>
      </div>

      {showAdd && <AddTextureForm pwd={pwd} onDone={() => { setShowAdd(false); load(); }} />}

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[0,1,2].map(i => <div key={i} className="h-40 rounded-xl animate-shimmer" />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {textures.map(t => {
            const isEditing = editingId === t.id;
            return (
              <div key={t.id} className={`bg-white rounded-xl border overflow-hidden shadow-xs transition-all ${isEditing ? "border-[#A01B1B]/40 ring-1 ring-[#A01B1B]/20" : "border-stone-200"}`}>
                {t.has_albedo && <img src={`/api/textures/${t.id}`} alt={t.name} className="h-28 w-full object-cover" />}
                <div className="p-3">
                  {isEditing ? (
                    <div className="space-y-2">
                      {META_FIELDS.map(f => (
                        <div key={f.key}>
                          <label className="text-[9px] font-medium text-stone-500 uppercase tracking-wide">{f.label}</label>
                          <input type={f.type} step={f.step} value={String(editMeta[f.key] ?? "")} onChange={e => setField(f.key, e.target.value, f.type)} className="w-full px-2.5 py-1.5 text-xs border border-stone-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-[#A01B1B]/50" />
                        </div>
                      ))}
                      <div>
                        <label className="text-[9px] font-medium text-stone-500 uppercase tracking-wide block mb-1">Tagi</label>
                        <div className="flex flex-wrap gap-1.5">
                          {ALL_TAGS.map(tag => {
                            const active = Array.isArray(editMeta.tags) && editMeta.tags.includes(tag);
                            return (
                              <button key={tag} type="button" onClick={() => toggleTag(tag)} className={`px-2 py-0.5 rounded-full text-[9px] font-medium border cursor-pointer transition-all ${active ? "text-white border-transparent" : "text-stone-500 border-stone-200 hover:border-stone-400"}`} style={active ? { backgroundColor: RED } : {}}>
                                {tag}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                      <div className="flex gap-2 pt-1">
                        <button onClick={handleSave} disabled={saving} className="px-3 py-1.5 text-[10px] font-semibold text-white rounded-lg disabled:opacity-50 cursor-pointer" style={{ backgroundColor: RED }}>
                          {saving ? "…" : "Zapisz"}
                        </button>
                        <button onClick={cancelEdit} className="px-3 py-1.5 text-[10px] text-stone-500 cursor-pointer">Anuluj</button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <p className="text-xs font-semibold text-stone-700 truncate">{t.name}</p>
                      <p className="text-[10px] text-stone-400">{t.id}</p>
                      {Array.isArray(t.meta.tags) && t.meta.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {t.meta.tags.map(tag => (
                            <span key={tag} className="px-1.5 py-0.5 rounded text-[8px] font-medium bg-stone-100 text-stone-500">{tag}</span>
                          ))}
                        </div>
                      )}
                      <div className="text-[9px] text-stone-400 mt-1">
                        {t.meta.moduleWidthMm && <span className="mr-2">{t.meta.moduleWidthMm}×{t.meta.moduleHeightMm}mm</span>}
                        {t.meta.jointMm !== undefined && <span className="mr-2">fuga {t.meta.jointMm}mm</span>}
                        {t.meta.layoutType && <span>{t.meta.layoutType}</span>}
                      </div>
                      <div className="flex gap-2 mt-2">
                        <button onClick={() => startEdit(t)} className="text-[10px] font-medium hover:underline cursor-pointer" style={{ color: RED }}>Edytuj</button>
                        <button onClick={() => handleDelete(t.id)} className="text-[10px] text-red-400 hover:text-red-600 cursor-pointer">Usuń</button>
                      </div>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function AddTextureForm({ pwd, onDone }: { pwd: string; onDone: () => void }) {
  const [id, setId] = useState("");
  const [meta, setMeta] = useState<TextureMeta>({ name: "", moduleWidthMm: 245, moduleHeightMm: 80, jointMm: 10, layoutType: "running-bond", offsetRatio: 0.5, tags: [] });
  const [preview, setPreview] = useState<string | null>(null);
  const [b64, setB64] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return;
    const reader = new FileReader();
    reader.onload = () => { const r = reader.result as string; setB64(r); setPreview(r); };
    reader.readAsDataURL(f);
  };

  const setField = (key: string, val: string, type: string) => {
    setMeta(prev => ({ ...prev, [key]: type === "number" ? (val === "" ? undefined : Number(val)) : val }));
  };

  const toggleTag = (tag: string) => {
    setMeta(prev => {
      const c = Array.isArray(prev.tags) ? prev.tags : [];
      return { ...prev, tags: c.includes(tag) ? c.filter(t => t !== tag) : [...c, tag] };
    });
  };

  const handleSave = async () => {
    setError("");
    if (!id || !meta.name || !b64) { setError("ID, nazwa i obraz są wymagane"); return; }
    setSaving(true);
    try {
      const r = await api("/textures", pwd, { method: "POST", body: JSON.stringify({ id, albedo_base64: b64, ...meta }) });
      if (!r.ok) { const d = await r.json().catch(() => ({})); setError(d.detail || "Błąd"); setSaving(false); return; }
      onDone();
    } catch { setError("Błąd sieci"); }
    setSaving(false);
  };

  return (
    <div className="bg-stone-50 rounded-xl border border-stone-200 p-4 mb-4 animate-fade-in">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <div>
          <label className="text-[9px] font-medium text-stone-500 uppercase tracking-wide block mb-1">ID (slug)</label>
          <input value={id} onChange={e => setId(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "-"))} className="w-full px-3 py-2 text-xs border rounded-lg" placeholder="moja-tekstura" />
        </div>
        {META_FIELDS.map(f => (
          <div key={f.key}>
            <label className="text-[9px] font-medium text-stone-500 uppercase tracking-wide block mb-1">{f.label}</label>
            <input type={f.type} step={f.step} value={String(meta[f.key] ?? "")} onChange={e => setField(f.key, e.target.value, f.type)} className="w-full px-3 py-2 text-xs border rounded-lg" />
          </div>
        ))}
        <div>
          <label className="text-[9px] font-medium text-stone-500 uppercase tracking-wide block mb-1">Obraz</label>
          <button onClick={() => fileRef.current?.click()} className="w-full px-3 py-2 text-xs border border-dashed rounded-lg text-stone-500 hover:border-[#A01B1B]/50 cursor-pointer">{preview ? "Zmień" : "Wybierz"}</button>
          <input ref={fileRef} type="file" accept="image/*" onChange={handleFile} className="hidden" />
        </div>
      </div>
      <div className="mt-3">
        <label className="text-[9px] font-medium text-stone-500 uppercase tracking-wide block mb-1">Tagi</label>
        <div className="flex flex-wrap gap-1.5">
          {ALL_TAGS.map(tag => {
            const active = Array.isArray(meta.tags) && meta.tags.includes(tag);
            return (
              <button key={tag} type="button" onClick={() => toggleTag(tag)} className={`px-2 py-0.5 rounded-full text-[9px] font-medium border cursor-pointer transition-all ${active ? "text-white border-transparent" : "text-stone-500 border-stone-200"}`} style={active ? { backgroundColor: RED } : {}}>
                {tag}
              </button>
            );
          })}
        </div>
      </div>
      {preview && <img src={preview} alt="Preview" className="w-16 h-16 mt-3 rounded-lg object-cover border" />}
      {error && <p className="text-xs text-red-600 mt-2">{error}</p>}
      <div className="flex gap-2 mt-3">
        <button onClick={handleSave} disabled={saving} className="px-4 py-2 text-xs font-semibold text-white rounded-lg disabled:opacity-50 cursor-pointer" style={{ backgroundColor: RED }}>{saving ? "…" : "Dodaj"}</button>
        <button onClick={onDone} className="px-4 py-2 text-xs text-stone-500 cursor-pointer">Anuluj</button>
      </div>
    </div>
  );
}

// ── Generations Gallery ──────────────────────────────────────────────────────

function GenerationsPanel({ pwd }: { pwd: string }) {
  const [items, setItems] = useState<GenItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [filterIp, setFilterIp] = useState("");
  const [lightbox, setLightbox] = useState<string | null>(null);
  const limit = 30;

  const load = useCallback(async (off: number) => {
    setLoading(true);
    const params = new URLSearchParams({ limit: String(limit), offset: String(off) });
    if (filterIp) params.set("client_ip", filterIp);
    try {
      const r = await api(`/generations?${params}`, pwd);
      if (r.ok) { const d = await r.json(); setItems(d.items); setTotal(d.total); setOffset(off); }
    } catch {}
    setLoading(false);
  }, [pwd, filterIp]);

  useEffect(() => { load(0); }, [load]);

  const hasNext = offset + limit < total;
  const hasPrev = offset > 0;

  return (
    <div className="animate-fade-in">
      <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
        <h2 className="text-lg font-bold text-stone-800">Generacje ({total})</h2>
        <div className="flex items-center gap-2">
          <input
            type="text" value={filterIp}
            onChange={e => setFilterIp(e.target.value)}
            placeholder="Filtruj po IP…"
            className="px-3 py-1.5 text-xs border rounded-lg w-36"
          />
          <button onClick={() => load(0)} className="px-3 py-1.5 text-xs font-semibold text-white rounded-lg cursor-pointer" style={{ backgroundColor: RED }}>Szukaj</button>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {[0,1,2,3,4].map(i => <div key={i} className="aspect-square rounded-xl animate-shimmer" />)}
        </div>
      ) : items.length === 0 ? (
        <p className="text-xs text-stone-400 text-center py-10">Brak generacji</p>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {items.map(g => (
              <div key={g.id} className="bg-white rounded-xl border border-stone-200 overflow-hidden shadow-xs group">
                {g.has_image ? (
                  <img
                    src={`${API}/api/admin/generations/${g.id}/image`}
                    alt={g.product_name}
                    className="aspect-square w-full object-cover cursor-pointer hover:opacity-90 transition"
                    onClick={() => setLightbox(`${API}/api/admin/generations/${g.id}/image`)}
                    loading="lazy"
                  />
                ) : (
                  <div className="aspect-square bg-stone-100 flex items-center justify-center">
                    <span className="text-xs text-stone-400">Brak</span>
                  </div>
                )}
                <div className="p-2">
                  <p className="text-[10px] font-semibold text-stone-700 truncate">{g.product_name}</p>
                  <p className="text-[9px] text-stone-400">{g.timestamp}</p>
                  <p className="text-[9px] text-stone-400 font-mono">{g.client_ip}</p>
                  <p className="text-[8px] text-stone-300">{g.gemini_model} · {g.timings?.total ?? "?"}s</p>
                </div>
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between mt-4">
            <button onClick={() => load(offset - limit)} disabled={!hasPrev} className="px-3 py-1.5 text-xs font-medium text-stone-600 border rounded-lg disabled:opacity-30 cursor-pointer">← Poprzednie</button>
            <span className="text-xs text-stone-400">{offset + 1}–{Math.min(offset + limit, total)} z {total}</span>
            <button onClick={() => load(offset + limit)} disabled={!hasNext} className="px-3 py-1.5 text-xs font-medium text-stone-600 border rounded-lg disabled:opacity-30 cursor-pointer">Następne →</button>
          </div>
        </>
      )}

      {/* Lightbox */}
      {lightbox && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setLightbox(null)}>
          <img src={lightbox} alt="Generacja" className="max-w-[90vw] max-h-[90vh] rounded-2xl shadow-2xl" onClick={e => e.stopPropagation()} />
          <button onClick={() => setLightbox(null)} className="absolute top-4 right-4 w-10 h-10 rounded-full bg-white/10 flex items-center justify-center cursor-pointer hover:bg-white/20 transition">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
          </button>
        </div>
      )}
    </div>
  );
}

// ── Watermark Panel ──────────────────────────────────────────────────────────

function WatermarkPanel({ pwd }: { pwd: string }) {
  const [config, setConfig] = useState<WmConfig | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try { const r = await api("/watermark", pwd); if (r.ok) setConfig(await r.json()); } catch {}
  }, [pwd]);
  useEffect(() => { load(); }, [load]);

  const handleToggle = async () => { if (!config) return; setSaving(true); await api("/watermark", pwd, { method: "PUT", body: JSON.stringify({ enabled: !config.enabled }) }); await load(); setSaving(false); };
  const handleOpacity = async (val: number) => { setSaving(true); await api("/watermark", pwd, { method: "PUT", body: JSON.stringify({ opacity: val }) }); await load(); setSaving(false); };
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return; setSaving(true);
    const reader = new FileReader();
    reader.onload = async () => { const b64 = reader.result as string; setPreview(b64); await api("/watermark/upload", pwd, { method: "POST", body: JSON.stringify({ image_base64: b64 }) }); await load(); setSaving(false); };
    reader.readAsDataURL(f);
  };
  const handleDelete = async () => { if (!confirm("Usunąć watermark?")) return; await api("/watermark", pwd, { method: "DELETE" }); setPreview(null); await load(); };

  return (
    <div className="animate-fade-in max-w-xl">
      <h2 className="text-lg font-bold text-stone-800 mb-4">Watermark</h2>
      <div className="bg-white rounded-xl border border-stone-200 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-stone-700">Aktywny</p>
            <p className="text-xs text-stone-400">Watermark w prawym dolnym rogu</p>
          </div>
          <button onClick={handleToggle} disabled={saving} className={`w-12 h-7 rounded-full transition-colors cursor-pointer ${config?.enabled ? "" : "bg-stone-300"}`} style={config?.enabled ? { backgroundColor: RED } : {}}>
            <div className={`w-5 h-5 bg-white rounded-full shadow-sm transition-transform mx-1 ${config?.enabled ? "translate-x-5" : ""}`} />
          </button>
        </div>
        <div>
          <label className="text-xs font-medium text-stone-600">Przezroczystość: {((config?.opacity ?? 0.3) * 100).toFixed(0)}%</label>
          <input type="range" min="5" max="100" value={(config?.opacity ?? 0.3) * 100} onChange={e => handleOpacity(Number(e.target.value) / 100)} className="w-full mt-1" style={{ accentColor: RED }} />
        </div>
        <div>
          <p className="text-xs font-medium text-stone-600 mb-2">Obraz (PNG)</p>
          {(config?.has_file || preview) && (
            <div className="flex items-center gap-3 mb-2">
              <img src={preview || `${API}/api/admin/watermark/preview?t=${Date.now()}`} alt="Watermark" className="h-12 rounded border bg-stone-100 p-1" />
              <button onClick={handleDelete} className="text-[10px] text-red-500 hover:text-red-700 cursor-pointer">Usuń</button>
            </div>
          )}
          <button onClick={() => fileRef.current?.click()} className="px-4 py-2 text-xs border border-dashed rounded-lg text-stone-500 hover:border-[#A01B1B]/50 cursor-pointer">{config?.has_file ? "Zmień" : "Wgraj"}</button>
          <input ref={fileRef} type="file" accept="image/png" onChange={handleUpload} className="hidden" />
        </div>
      </div>
    </div>
  );
}

// ── Limits Panel ─────────────────────────────────────────────────────────────

function LimitsPanel({ pwd }: { pwd: string }) {
  const [data, setData] = useState<LimitsData | null>(null);
  const [newLimit, setNewLimit] = useState(50);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try { const r = await api("/limits", pwd); if (r.ok) { const d = await r.json(); setData(d); setNewLimit(d.daily_limit); } } catch {}
  }, [pwd]);
  useEffect(() => { load(); }, [load]);

  const handleSave = async () => { setSaving(true); await api("/limits", pwd, { method: "PUT", body: JSON.stringify({ daily_limit: newLimit }) }); await load(); setSaving(false); };

  const users = data?.usage?.users ?? {};
  const entries = Object.entries(users).sort((a, b) => b[1] - a[1]);

  return (
    <div className="animate-fade-in max-w-xl">
      <h2 className="text-lg font-bold text-stone-800 mb-4">Limity generacji</h2>
      <div className="bg-white rounded-xl border border-stone-200 p-5 space-y-4">
        <div>
          <label className="text-xs font-medium text-stone-600">Dzienny limit na użytkownika (0 = bez limitu)</label>
          <div className="flex gap-2 mt-1">
            <input type="number" min="0" value={newLimit} onChange={e => setNewLimit(Number(e.target.value))} className="w-24 px-3 py-2 text-sm border rounded-lg" />
            <button onClick={handleSave} disabled={saving} className="px-4 py-2 text-xs font-semibold text-white rounded-lg disabled:opacity-50 cursor-pointer" style={{ backgroundColor: RED }}>Zapisz</button>
          </div>
          <p className="text-[10px] text-stone-400 mt-1">Aktualny: {data?.daily_limit ?? "…"}/dobę</p>
        </div>
        <div>
          <p className="text-xs font-medium text-stone-600 mb-2">Dzisiejsze użycie ({data?.usage?.date ?? "…"})</p>
          {entries.length === 0 ? <p className="text-xs text-stone-400">Brak</p> : (
            <div className="space-y-1">
              {entries.map(([ip, count]) => (
                <div key={ip} className="flex items-center justify-between text-xs py-1.5 px-3 bg-stone-50 rounded-lg">
                  <span className="font-mono text-stone-500 text-[10px]">{ip}</span>
                  <span className="font-semibold text-stone-700">{count} / {data?.daily_limit}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
