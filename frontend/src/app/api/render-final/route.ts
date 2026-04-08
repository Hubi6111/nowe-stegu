import { NextRequest, NextResponse } from "next/server";
import sharp from "sharp";
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";

export const maxDuration = 60;

interface Point {
  x: number;
  y: number;
}

const WALL_HEIGHT_MM = 2850;
const MAX_PROCESS_DIM = 2048;
const MAX_TILES = 400;

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

function computeTextureScale(
  polyHeightPx: number,
  imageHeightPx: number,
  textureHeightPx: number,
  meta: Record<string, unknown>
): number {
  const moduleH = Number(meta.moduleHeightMm || 65);
  const joint = Number(meta.jointMm || 10);
  const layout = String(meta.layoutType || "running-bond");
  let courses = Number(meta.albedoBrickCourses || meta.albedoCourses || 8);
  courses = Math.max(1, Math.min(courses, 32));

  let albedoHMm: number;
  if (layout === "running-bond" || layout === "stretcher-bond") {
    albedoHMm = moduleH * courses + joint * Math.max(courses - 1, 0);
  } else {
    const planks = Math.max(
      1,
      Math.min(
        Number(meta.albedoStackPlanks || meta.albedoPlankCount || 2),
        12
      )
    );
    albedoHMm = moduleH * planks + joint * Math.max(planks - 1, 0);
  }

  const frac = polyHeightPx / imageHeightPx;
  const wallMm =
    frac >= 0.5
      ? WALL_HEIGHT_MM
      : Math.max(950, WALL_HEIGHT_MM * Math.max(frac / 0.5, 0.36));
  const pxPerMm = polyHeightPx / wallMm;
  const targetHPx = Math.max(albedoHMm * pxPerMm, 1);
  const scale = targetHPx / textureHeightPx;

  const mult = Math.max(
    0.35,
    Math.min(Number(meta.textureScaleMultiplier || 1), 2.5)
  );
  return scale * mult;
}

function buildRenderPrompt(
  productName: string,
  meta: Record<string, unknown>
): string {
  const moduleH = Number(meta.moduleHeightMm || 80);
  const moduleW = Number(meta.moduleWidthMm || 245);
  const joint = Number(meta.jointMm || 10);
  const layout = String(meta.layoutType || "running-bond");
  const isBrick = layout === "running-bond" || layout === "stretcher-bond";

  return [
    "You are a photorealistic interior rendering engine.",
    "",
    "You receive FOUR images:",
    "1. ORIGINAL — the unmodified room photo",
    "2. COMPOSITE — room with texture algorithmically placed on the wall",
    "3. MASK OVERLAY — original with ORANGE highlight showing where texture goes",
    "4. PRODUCT TEXTURE TILE — the actual decorative material",
    "",
    "YOUR TASK:",
    "Apply the texture ONLY within the orange-highlighted wall area and make it look like a real photograph of a renovated room.",
    "",
    "RENDERING RULES:",
    "• Texture goes ONLY on flat wall surfaces within the orange zone",
    "• Do NOT texture ceiling, floor, adjacent walls, or any objects/furniture",
    "• Match the room's existing lighting, color temperature, and exposure",
    "• Add realistic shadows at ceiling/floor/furniture junctions",
    `• Add surface relief consistent with real ${isBrick ? "brick" : "decorative"} material`,
    `• Each module: ${moduleW}mm × ${moduleH}mm, joint: ${joint}mm`,
    "• Preserve the composite's tiling pattern and scale",
    "• Everything outside the textured area must be identical to ORIGINAL",
    "",
    "RESTRICTIONS:",
    "• Do NOT zoom, crop, pan, or reframe",
    "• Do NOT change texture scale or pattern",
    "• Do NOT cover furniture or wall objects",
    "• Do NOT extend texture beyond the orange zone",
    "",
    `PRODUCT: ${productName}`,
    "",
    "Return ONLY the final photorealistic image. No text.",
  ].join("\n");
}

export async function POST(req: NextRequest) {
  const t0 = Date.now();
  const timings: Record<string, number> = {};

  try {
    const body = await req.json();
    const { image, polygon, product_id, canvas_width, canvas_height } = body as {
      image: string;
      polygon: Point[];
      product_id: string;
      canvas_width: number;
      canvas_height: number;
    };

    if (!image || !polygon || polygon.length < 2 || !product_id) {
      return NextResponse.json(
        { error: "image, polygon (≥2 points), and product_id are required" },
        { status: 400 }
      );
    }

    const imgBase64 = image.includes(",") ? image.split(",")[1] : image;
    const imgBuffer = Buffer.from(imgBase64, "base64");
    const imgMeta = await sharp(imgBuffer).metadata();
    const origW = imgMeta.width!;
    const origH = imgMeta.height!;

    const longest = Math.max(origW, origH);
    const downscale = longest > MAX_PROCESS_DIM ? MAX_PROCESS_DIM / longest : 1;
    const W = Math.round(origW * downscale);
    const H = Math.round(origH * downscale);

    const resizedImgBuffer =
      downscale < 1
        ? await sharp(imgBuffer).resize(W, H).jpeg({ quality: 92 }).toBuffer()
        : await sharp(imgBuffer).jpeg({ quality: 92 }).toBuffer();

    const sx = (W / canvas_width);
    const sy = (H / canvas_height);
    const scaledPoly = polygon.map((p) => ({
      x: p.x * sx,
      y: p.y * sy,
    }));

    const texturesDir = getTexturesDir();
    if (!texturesDir) {
      return NextResponse.json(
        { error: "Textures directory not found" },
        { status: 500 }
      );
    }

    const safeId = product_id.replace(/[^a-zA-Z0-9_-]/g, "");
    const texturePath = path.join(texturesDir, safeId, "albedo.jpg");
    const metaPath = path.join(texturesDir, safeId, "metadata.json");

    if (!fs.existsSync(texturePath) || !fs.existsSync(metaPath)) {
      return NextResponse.json(
        { error: `Product not found: ${safeId}` },
        { status: 404 }
      );
    }

    const textureFileBuffer = fs.readFileSync(texturePath);
    const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));

    // --- Texture tiling ---
    const t1 = Date.now();
    const texMeta = await sharp(textureFileBuffer).metadata();
    const polyYs = scaledPoly.map((p) => p.y);
    const polyHeight = Math.max(...polyYs) - Math.min(...polyYs);
    const texScale = computeTextureScale(polyHeight, H, texMeta.height!, meta);

    let tileW = Math.max(Math.round(texMeta.width! * texScale), 1);
    let tileH = Math.max(Math.round(texMeta.height! * texScale), 1);

    let tilesX = Math.ceil(W / tileW);
    let tilesY = Math.ceil(H / tileH);
    if (tilesX * tilesY > MAX_TILES) {
      const factor = Math.sqrt((tilesX * tilesY) / MAX_TILES);
      tileW = Math.round(tileW * factor);
      tileH = Math.round(tileH * factor);
      tilesX = Math.ceil(W / tileW);
      tilesY = Math.ceil(H / tileH);
    }

    const scaledTexBuf = await sharp(textureFileBuffer)
      .resize(tileW, tileH)
      .toBuffer();

    const tileOps: sharp.OverlayOptions[] = [];
    for (let y = 0; y < H; y += tileH) {
      for (let x = 0; x < W; x += tileW) {
        tileOps.push({ input: scaledTexBuf, left: x, top: y });
      }
    }

    const tiledBuffer = await sharp({
      create: {
        width: W,
        height: H,
        channels: 4,
        background: { r: 0, g: 0, b: 0, alpha: 255 },
      },
    })
      .composite(tileOps)
      .png()
      .toBuffer();

    // --- Polygon mask ---
    const points = scaledPoly
      .map((p) => `${Math.round(p.x)},${Math.round(p.y)}`)
      .join(" ");

    const maskSvg = Buffer.from(
      `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">` +
        `<polygon points="${points}" fill="white"/>` +
        `</svg>`
    );
    const maskPng = await sharp(maskSvg).ensureAlpha().png().toBuffer();

    const maskedTexture = await sharp(tiledBuffer)
      .composite([{ input: maskPng, blend: "dest-in" }])
      .png()
      .toBuffer();

    const compositeBuffer = await sharp(resizedImgBuffer)
      .composite([{ input: maskedTexture }])
      .jpeg({ quality: 92 })
      .toBuffer();

    const compositeB64 =
      "data:image/jpeg;base64," + compositeBuffer.toString("base64");
    timings.texture_project = Math.round((Date.now() - t1) / 100) / 10;

    // --- Mask overlay (orange highlight for Gemini) ---
    const overlaySvg = Buffer.from(
      `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">` +
        `<polygon points="${points}" fill="rgba(255,140,0,0.45)"/>` +
        `</svg>`
    );
    const overlayPng = await sharp(overlaySvg).png().toBuffer();
    const maskOverlayBuffer = await sharp(resizedImgBuffer)
      .composite([{ input: overlayPng }])
      .jpeg({ quality: 85 })
      .toBuffer();

    // --- Gemini photorealistic render ---
    const t2 = Date.now();
    const apiKey = process.env.GEMINI_API_KEY;
    let refinedB64 = compositeB64;
    let geminiModel = "not-configured";

    if (apiKey) {
      try {
        const modelId =
          process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image-preview";
        geminiModel = modelId;

        const genAI = new GoogleGenerativeAI(apiKey);
        const model = genAI.getGenerativeModel({
          model: modelId,
          generationConfig: { temperature: 0.2 },
        });

        const prompt = buildRenderPrompt(meta.name || product_id, meta);

        const result = await model.generateContent([
          { text: prompt },
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: resizedImgBuffer.toString("base64"),
            },
          },
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: compositeBuffer.toString("base64"),
            },
          },
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: maskOverlayBuffer.toString("base64"),
            },
          },
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: textureFileBuffer.toString("base64"),
            },
          },
        ]);

        const parts = result.response.candidates?.[0]?.content?.parts;
        const imagePart = parts?.find(
          (p: { inlineData?: { mimeType: string; data: string } }) =>
            p.inlineData
        );

        if (imagePart?.inlineData) {
          refinedB64 = `data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`;
        }
      } catch (e) {
        console.error("[render-final] Gemini error:", e);
        geminiModel = `error: ${e instanceof Error ? e.message : String(e)}`;
      }
    }

    timings.gemini_render = Math.round((Date.now() - t2) / 100) / 10;
    timings.total = Math.round((Date.now() - t0) / 100) / 10;

    return NextResponse.json({
      composite: compositeB64,
      refined: refinedB64,
      gemini_model: geminiModel,
      timings,
    });
  } catch (e) {
    console.error("[render-final]", e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Render failed" },
      { status: 500 }
    );
  }
}
