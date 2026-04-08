import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";
import fs from "fs";
import path from "path";

const DEFAULT_HASH = crypto
  .createHash("sha256")
  .update("stegu")
  .digest("hex");

function getPasswordHash(): string {
  const configPaths = [
    path.join(process.cwd(), "data", "admin-config.json"),
    path.join(process.cwd(), "..", "data", "admin-config.json"),
  ];
  for (const p of configPaths) {
    try {
      if (fs.existsSync(p)) {
        const cfg = JSON.parse(fs.readFileSync(p, "utf-8"));
        if (cfg.admin_password_hash) return cfg.admin_password_hash;
      }
    } catch {}
  }
  return DEFAULT_HASH;
}

function verifyPassword(password: string): boolean {
  const hash = crypto.createHash("sha256").update(password).digest("hex");
  return hash === getPasswordHash();
}

function requireAdmin(req: NextRequest): NextResponse | null {
  const pwd = req.headers.get("x-admin-password") || "";
  if (!verifyPassword(pwd)) {
    return NextResponse.json(
      { detail: "Nieprawidłowe hasło" },
      { status: 401 }
    );
  }
  return null;
}

function getTexturesDir(): string | null {
  const candidates = [
    path.join(process.cwd(), "assets", "textures", "stegu"),
    path.join(process.cwd(), "..", "assets", "textures", "stegu"),
    path.join(process.cwd(), "frontend", "assets", "textures", "stegu"),
  ];
  for (const dir of candidates) {
    if (fs.existsSync(dir)) return dir;
  }
  return null;
}

function listTextures() {
  const dir = getTexturesDir();
  if (!dir) return [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  return entries
    .filter((e) => e.isDirectory())
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((folder) => {
      const metaPath = path.join(dir, folder.name, "metadata.json");
      const hasAlbedo = fs.existsSync(
        path.join(dir, folder.name, "albedo.jpg")
      );
      let meta: Record<string, unknown> = {};
      try {
        if (fs.existsSync(metaPath))
          meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
      } catch {}
      return {
        id: folder.name,
        name: (meta.name as string) || folder.name,
        has_albedo: hasAlbedo,
        meta,
      };
    });
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: segments } = await params;
  const route = segments.join("/");

  if (route === "login") {
    try {
      const body = await req.json();
      if (!verifyPassword(body.password || "")) {
        return NextResponse.json(
          { detail: "Nieprawidłowe hasło" },
          { status: 401 }
        );
      }
      return NextResponse.json({ ok: true });
    } catch {
      return NextResponse.json({ detail: "Invalid request" }, { status: 400 });
    }
  }

  const authErr = requireAdmin(req);
  if (authErr) return authErr;

  if (route === "change-password") {
    return NextResponse.json(
      { detail: "Zmiana hasła niedostępna na Vercel" },
      { status: 501 }
    );
  }

  if (route === "textures") {
    return NextResponse.json(
      { detail: "Dodawanie tekstur niedostępne na Vercel" },
      { status: 501 }
    );
  }

  if (route === "watermark/upload") {
    return NextResponse.json(
      { detail: "Upload watermarku niedostępny na Vercel" },
      { status: 501 }
    );
  }

  return NextResponse.json({ detail: "Not found" }, { status: 404 });
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path: segments } = await params;
  const route = segments.join("/");

  if (route === "watermark/preview") {
    return NextResponse.json({ detail: "Brak watermarku" }, { status: 404 });
  }

  const authErr = requireAdmin(req);
  if (authErr) return authErr;

  if (route === "textures") {
    return NextResponse.json(listTextures());
  }

  if (route === "stats") {
    return NextResponse.json({
      total: 0,
      today: 0,
      unique_ips: 0,
      by_product: {},
      by_day: {},
      daily_limit: 0,
      usage_today: { date: new Date().toISOString().slice(0, 10), users: {}, total: 0 },
    });
  }

  if (route === "generations") {
    return NextResponse.json({
      total: 0,
      offset: 0,
      limit: 50,
      items: [],
    });
  }

  if (route === "watermark") {
    return NextResponse.json({
      enabled: false,
      opacity: 0.3,
      has_file: false,
    });
  }

  if (route === "limits") {
    return NextResponse.json({
      daily_limit: 0,
      usage: { date: new Date().toISOString().slice(0, 10), users: {}, total: 0 },
    });
  }

  return NextResponse.json({ detail: "Not found" }, { status: 404 });
}

export async function PUT(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const authErr = requireAdmin(req);
  if (authErr) return authErr;

  const { path: segments } = await params;
  const route = segments.join("/");

  if (route === "watermark") {
    return NextResponse.json({ enabled: false, opacity: 0.3, has_file: false });
  }

  if (route === "limits") {
    return NextResponse.json({ daily_limit: 0 });
  }

  if (route.startsWith("textures/")) {
    return NextResponse.json(
      { detail: "Edycja tekstur niedostępna na Vercel" },
      { status: 501 }
    );
  }

  return NextResponse.json({ detail: "Not found" }, { status: 404 });
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const authErr = requireAdmin(req);
  if (authErr) return authErr;

  return NextResponse.json(
    { detail: "Usuwanie niedostępne na Vercel" },
    { status: 501 }
  );
}
