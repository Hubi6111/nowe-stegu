import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

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

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ productId: string }> }
) {
  const { productId } = await params;

  if (/[^a-zA-Z0-9_-]/.test(productId)) {
    return NextResponse.json({ error: "Invalid product ID" }, { status: 400 });
  }

  const texturesDir = getTexturesDir();
  if (!texturesDir) {
    return NextResponse.json({ error: "Textures directory not found" }, { status: 404 });
  }

  const filePath = path.join(texturesDir, productId, "albedo.jpg");
  if (!fs.existsSync(filePath)) {
    return NextResponse.json({ error: "Texture not found" }, { status: 404 });
  }

  const buffer = fs.readFileSync(filePath);
  return new NextResponse(buffer, {
    headers: {
      "Content-Type": "image/jpeg",
      "Cache-Control": "public, max-age=86400, immutable",
    },
  });
}
