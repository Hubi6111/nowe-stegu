import { NextResponse } from "next/server";

function normalizeApiBase(url: string): string {
  const u = url.replace(/\/$/, "");
  // Prefer IPv4 loopback — fewer ECONNRESET issues than localhost → ::1 on macOS.
  return u.replace(/^http:\/\/localhost\b/, "http://127.0.0.1");
}

/** Proxies FastAPI /api/health; returns safe JSON if backend is unreachable (no socket hang up). */
export async function GET() {
  const raw =
    process.env.API_PROXY_URL ||
    process.env.API_BASE_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "";
  const base = raw ? normalizeApiBase(raw) : "";

  if (base) {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 10000);
      const r = await fetch(`${base}/api/health`, {
        cache: "no-store",
        signal: ctrl.signal,
        headers: { Connection: "close" },
      });
      clearTimeout(t);
      if (r.ok) {
        const data = (await r.json()) as Record<string, unknown>;
        return NextResponse.json(data);
      }
    } catch {
      /* backend down or reset — degrade gracefully */
    }
  }

  const geminiKey = !!process.env.GEMINI_API_KEY;
  return NextResponse.json({
    status: geminiKey ? "ok-standalone" : "degraded",
    gemini_configured: geminiKey,
    mode: "nextjs-standalone",
    inference_url: base || "none",
    inference_reachable: false,
    note: geminiKey
      ? "Running in Next.js standalone mode with Gemini API"
      : "FastAPI unreachable and GEMINI_API_KEY not set",
  });
}
