import { NextRequest, NextResponse } from "next/server";
import sharp from "sharp";
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { getClientIp, checkRateLimit, recordUsage } from "../rate-limit";

export const maxDuration = 300;

interface Point { x: number; y: number }

const MAX_PROCESS_DIM = 2048;
const MAX_TILES = 400;
const WATERMARK_OPACITY = 0.3;
const WATERMARK_MAX_WIDTH_RATIO = 0.18;
const WATERMARK_MAX_HEIGHT_RATIO = 0.10;
const WATERMARK_MARGIN = 20;

/* ── Path helpers ── */

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

function getWatermarkPath(): string | null {
  const candidates = [
    path.join(process.cwd(), "public", "watermark.png"),
    path.join(process.cwd(), "..", "public", "watermark.png"),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

/* ══════════════════════════════════════════════════════════
   PHASE 1: AI Scene Dimension Analysis
   Uses reference objects to estimate real-world wall size
   ══════════════════════════════════════════════════════════ */

const SCENE_DIMENSION_PROMPT = `You are an expert at estimating real-world dimensions from photographs using reference objects.

You receive TWO images:
1. ORIGINAL room/building photograph
2. MASK OVERLAY — same photo with ORANGE highlight showing the wall area the user selected

YOUR TASK: Estimate the REAL-WORLD dimensions (in centimeters) of the ORANGE-highlighted wall area.

═══ REFERENCE OBJECTS WITH KNOWN DIMENSIONS ═══
Use ANY of these visible in the photo to triangulate dimensions:

DOORS & WINDOWS:
• Standard interior door height: 200 cm (THIS IS YOUR PRIMARY REFERENCE)
• Standard interior door width: 80-90 cm
• Standard exterior door: 200-210 cm tall × 90-100 cm wide
• Standard window height: 120-150 cm
• Window sill from floor: 80-90 cm

ROOM STRUCTURE:
• Typical European interior ceiling height: 250-280 cm
• Standard baseboard/skirting: 8-12 cm tall
• Standard door frame/architrave width: 6-8 cm

ELECTRICAL:
• Light switch plate: ~8×8 cm, mounted 110-120 cm from floor
• Power outlet plate: ~8×8 cm, mounted 25-30 cm from floor

FURNITURE:
• Dining table height: 75 cm
• Kitchen counter height: 85-90 cm
• Standard chair seat: 45 cm from floor
• Sofa seat height: 40-45 cm, back height: 80-90 cm
• Standard bookshelf: 180-200 cm tall

EXTERIOR:
• Standard brick course: 6.5-8 cm tall (if existing bricks visible)
• Standard window: 120×100 cm
• Garage door: 210-240 cm tall × 240-300 cm wide
• Fence post height: 100-180 cm

═══ METHOD ═══
1. Identify ALL reference objects visible in the image
2. For EACH reference, estimate how many pixels it occupies vs how many pixels the wall area occupies
3. Cross-reference multiple objects to increase accuracy
4. Account for perspective foreshortening (objects further from camera appear smaller)
5. For the highlighted wall area, estimate both height and width in real centimeters

═══ CRITICAL RULES ═══
• If a door is visible, it is ALWAYS 200 cm tall — use this as your primary anchor
• Interior ceiling is 250-280 cm unless clearly unusual
• Be conservative: slightly underestimate rather than overestimate
• Consider if the highlighted area is a PARTIAL wall (not floor-to-ceiling)

OUTPUT a single JSON object (no markdown, no code fences):
{"wallHeightCm":220,"wallWidthCm":300,"ceilingHeightCm":265,"sceneType":"interior","referenceObjects":[{"name":"door","estimatedSizeCm":200,"pixelsCovered":450,"confidence":"high"}],"confidence":"high","perspectiveAngle":"frontal","notes":"Door clearly visible at left edge, used as primary reference"}`;

interface SceneDimensions {
  wallHeightCm: number;
  wallWidthCm: number;
  ceilingHeightCm: number | null;
  sceneType: "interior" | "exterior";
  confidence: "high" | "medium" | "low";
  perspectiveAngle: string;
  referenceObjects: { name: string; estimatedSizeCm: number; confidence: string }[];
}

const DEFAULT_DIMENSIONS: SceneDimensions = {
  wallHeightCm: 260,
  wallWidthCm: 350,
  ceilingHeightCm: 270,
  sceneType: "interior",
  confidence: "low",
  perspectiveAngle: "frontal",
  referenceObjects: [],
};

async function analyzeSceneDimensions(
  genAI: GoogleGenerativeAI,
  originalBuf: Buffer,
  maskOverlayBuf: Buffer,
  polyHeightFraction: number,
  polyWidthFraction: number,
): Promise<SceneDimensions> {
  const textModelId = process.env.GEMINI_TEXT_MODEL || "gemini-2.5-pro";
  try {
    const model = genAI.getGenerativeModel({ model: textModelId });
    const result = await geminiRetry(() => model.generateContent([
      { text: SCENE_DIMENSION_PROMPT },
      { inlineData: { mimeType: "image/jpeg", data: originalBuf.toString("base64") } },
      { inlineData: { mimeType: "image/jpeg", data: maskOverlayBuf.toString("base64") } },
    ]));
    const text = result.response.text?.() ?? "";
    const clean = text.replace(/```(?:json)?\s*/g, "").replace(/```/g, "").trim();
    const m = clean.match(/\{[\s\S]*\}/);
    if (m) {
      const parsed = JSON.parse(m[0]);
      const dims: SceneDimensions = {
        wallHeightCm: clamp(Number(parsed.wallHeightCm) || 260, 30, 1200),
        wallWidthCm: clamp(Number(parsed.wallWidthCm) || 350, 30, 2000),
        ceilingHeightCm: parsed.ceilingHeightCm ? clamp(Number(parsed.ceilingHeightCm), 200, 500) : null,
        sceneType: parsed.sceneType === "exterior" ? "exterior" : "interior",
        confidence: (["high", "medium", "low"] as const).includes(parsed.confidence) ? parsed.confidence : "medium",
        perspectiveAngle: String(parsed.perspectiveAngle || "frontal"),
        referenceObjects: Array.isArray(parsed.referenceObjects) ? parsed.referenceObjects : [],
      };

      // Sanity checks using polygon fraction
      if (dims.sceneType === "interior") {
        const impliedCeiling = dims.wallHeightCm / polyHeightFraction;
        if (impliedCeiling > 500) {
          dims.wallHeightCm = Math.round(280 * polyHeightFraction * 1.1);
          dims.confidence = "low";
        }
        if (impliedCeiling < 180 && polyHeightFraction < 0.9) {
          dims.wallHeightCm = Math.round(265 * polyHeightFraction);
          dims.confidence = "low";
        }
      }

      console.log("[render-final] AI dimensions:", JSON.stringify(dims));
      return dims;
    }
  } catch (e) {
    console.warn("[render-final] Scene dimension analysis failed:", e);
  }

  // Fallback: heuristic estimation
  const fallbackCeiling = 270;
  return {
    ...DEFAULT_DIMENSIONS,
    wallHeightCm: Math.round(fallbackCeiling * polyHeightFraction * 1.05),
    wallWidthCm: Math.round(fallbackCeiling * polyWidthFraction * 1.05),
  };
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/* ══════════════════════════════════════════════════════════
   PHASE 2: Precise texture scale computation
   Uses AI dimensions + product physical specs
   ══════════════════════════════════════════════════════════ */

interface ScaleReport {
  scale: number;
  wallHeightMm: number;
  wallWidthMm: number;
  courseHeightMm: number;
  unitWidthMm: number;
  expectedCoursesH: number;
  expectedUnitsW: number;
  coursesInPoly: number;
  unitsInPolyRow: number;
  brickBodyHeightMm: number;
  brickBodyWidthMm: number;
  jointMm: number;
  dimensionSource: "ai" | "heuristic";
  verified: boolean;
}

function computeTextureScale(
  polyHeightPx: number,
  polyWidthPx: number,
  textureHeightPx: number,
  textureWidthPx: number,
  meta: Record<string, unknown>,
  sceneDims: SceneDimensions,
): ScaleReport {
  const moduleH = Number(meta.moduleHeightMm || 80);
  const moduleW = Number(meta.moduleWidthMm || 245);
  const joint = Number(meta.jointMm || 10);
  const layout = String(meta.layoutType || "running-bond");
  let courses = Number(meta.albedoBrickCourses || meta.albedoCourses || 8);
  courses = clamp(courses, 1, 32);
  const isBrick = layout === "running-bond" || layout === "stretcher-bond";

  const courseH = moduleH + joint;
  const unitW = moduleW + joint;

  // Wall dimensions from AI analysis (mm)
  const wallHMm = sceneDims.wallHeightCm * 10;
  const wallWMm = sceneDims.wallWidthCm * 10;

  // Physical size of one albedo tile
  let albedoHMm: number;
  let albedoWMm: number;
  if (isBrick) {
    albedoHMm = courseH * courses;
    const bricksPerRow = Math.max(1, Math.round(textureWidthPx / textureHeightPx * courses));
    albedoWMm = unitW * bricksPerRow;
  } else {
    const planks = clamp(Number(meta.albedoStackPlanks || meta.albedoPlankCount || 2), 1, 12);
    albedoHMm = (moduleH + joint) * planks;
    albedoWMm = moduleW;
  }

  // How many courses/units should fit in the wall
  const expectedCoursesH = wallHMm / courseH;
  const expectedUnitsW = wallWMm / unitW;

  // Compute pixel scale: how many pixels per mm of real-world wall
  const pxPerMm = polyHeightPx / wallHMm;
  const targetAlbedoHPx = albedoHMm * pxPerMm;
  const scale = targetAlbedoHPx / textureHeightPx;

  // Verify: count actual courses at this scale
  const tileHPx = textureHeightPx * scale;
  const coursesInPoly = (polyHeightPx / tileHPx) * courses;

  const tileWPx = textureWidthPx * scale;
  const albedoRepeatsW = polyWidthPx / tileWPx;
  const bricksPerAlbedo = isBrick ? Math.max(1, Math.round(textureWidthPx / textureHeightPx * courses)) : 1;
  const unitsInPolyRow = albedoRepeatsW * bricksPerAlbedo;

  const mult = clamp(Number(meta.textureScaleMultiplier || 1), 0.5, 2.0);
  const finalScale = Math.max(0.02, scale * mult);

  return {
    scale: finalScale,
    wallHeightMm: Math.round(wallHMm),
    wallWidthMm: Math.round(wallWMm),
    courseHeightMm: courseH,
    unitWidthMm: unitW,
    expectedCoursesH: Math.round(expectedCoursesH),
    expectedUnitsW: Math.round(expectedUnitsW),
    coursesInPoly: Math.round(coursesInPoly),
    unitsInPolyRow: Math.round(unitsInPolyRow),
    brickBodyHeightMm: moduleH,
    brickBodyWidthMm: moduleW,
    jointMm: joint,
    dimensionSource: sceneDims.confidence !== "low" ? "ai" : "heuristic",
    verified: Math.abs(coursesInPoly - expectedCoursesH) / expectedCoursesH < 0.15,
  };
}

/* ══════════════════════════════════════════════════════════
   (Lighting analysis is now integrated into the render prompt
    to reduce the number of sequential AI calls from 3 to 2)
   ══════════════════════════════════════════════════════════ */

/* ══════════════════════════════════════════════════════════
   Photorealistic render prompt (Phase 3)
   ══════════════════════════════════════════════════════════ */

function buildRenderPrompt(
  productName: string,
  meta: Record<string, unknown>,
  scaleReport: ScaleReport,
  sceneDims: SceneDimensions,
): string {
  const layout = String(meta.layoutType || "running-bond");
  const isBrick = layout === "running-bond" || layout === "stretcher-bond";

  return `You are a photorealistic compositing engine. You blend a pre-tiled wall texture into a room photo so the result looks like a real renovation photograph.

YOU RECEIVE 4 IMAGES:
1. ORIGINAL — unmodified room photo (your lighting/perspective reference)
2. COMPOSITE — the room with the EXACT product texture already tiled on the wall at correct scale
3. MASK — ORIGINAL with ORANGE overlay showing the wall area
4. TEXTURE TILE — the actual "${productName}" product (reference for exact color/pattern)

YOUR TASK — STRICTLY:
Take image 2 (COMPOSITE) as your starting point. The texture pattern, colors, brick layout, mortar lines, tile count, and scale in the COMPOSITE are already PERFECT and FINAL. DO NOT redraw, reimagine, or regenerate any part of the texture. Your ONLY job is to make the COMPOSITE blend seamlessly into the room by adding:

1. LIGHTING MATCH — Study image 1 (ORIGINAL). Match its light direction, color temperature, exposure, and shadows. Apply consistent lighting onto the textured wall so it feels lit by the same light source as the rest of the room.

2. SURFACE DEPTH — Add subtle ${isBrick ? "brick relief: slight highlight on top edges, shadow on bottom edges of individual bricks, recessed mortar joints with micro-shadows" : "panel relief: subtle edge shadows between panels"}. Keep this VERY subtle — real ${isBrick ? "brick cladding" : "wall panels"} on a wall has only millimeters of depth.

3. EDGE BLENDING — Where the texture meets non-textured areas (ceiling, floor, other walls, furniture), blend naturally. Soften the boundary so it doesn't look cut-and-pasted.

4. PERSPECTIVE — If the wall is at an angle to the camera, apply gentle perspective foreshortening to the texture so ${isBrick ? "mortar lines converge toward vanishing points" : "panel edges follow the wall plane"}.

5. AMBIENT OCCLUSION — Darken subtly where wall meets ceiling, floor, and corners.

6. CONTACT SHADOWS — Where furniture/objects touch the textured wall, add natural shadow.

TEXTURE FIDELITY IS THE #1 PRIORITY:
• The EXACT colors from image 4 (TEXTURE TILE) must appear in the final output — same hue, same saturation, same tone
• Every individual ${isBrick ? "brick pattern, color variation, surface crack, and mortar shade" : "panel grain, color variation, and surface detail"} visible in the COMPOSITE must remain UNCHANGED
• If you see ${scaleReport.coursesInPoly} ${isBrick ? "brick courses" : "rows"} in the COMPOSITE, your output must have EXACTLY ${scaleReport.coursesInPoly} ${isBrick ? "courses" : "rows"}
• The texture is a real commercial product — the customer MUST be able to recognize it as "${productName}"

ABSOLUTE RULES:
• Start from the COMPOSITE (image 2) — do NOT start from scratch
• Do NOT invent, hallucinate, or regenerate the texture — it's already there
• Do NOT change texture colors, pattern, scale, or tile count
• Do NOT alter anything outside the masked wall area — keep it pixel-identical to image 1
• Do NOT cover furniture, objects, windows, doors, or fixtures
• Do NOT zoom, crop, or change image dimensions
• Keep the same image resolution and aspect ratio

The wall is approximately ${sceneDims.wallHeightCm}cm × ${sceneDims.wallWidthCm}cm (${sceneDims.sceneType}). ${isBrick ? `Each brick is ${scaleReport.brickBodyWidthMm}×${scaleReport.brickBodyHeightMm}mm with ${scaleReport.jointMm}mm mortar.` : ""}

Output ONLY the final image. No text.`;
}

/* ── Watermark ── */

async function applyWatermark(imageBuffer: Buffer): Promise<Buffer> {
  const wmPath = getWatermarkPath();
  if (!wmPath) return imageBuffer;
  try {
    const imgMeta = await sharp(imageBuffer).metadata();
    const imgW = imgMeta.width!;
    const imgH = imgMeta.height!;
    const wmRaw = fs.readFileSync(wmPath);
    const wmMeta = await sharp(wmRaw).metadata();
    const wmW = wmMeta.width!;
    const wmH = wmMeta.height!;
    const maxW = Math.round(imgW * WATERMARK_MAX_WIDTH_RATIO);
    const maxH = Math.round(imgH * WATERMARK_MAX_HEIGHT_RATIO);
    const scale = Math.min(maxW / wmW, maxH / wmH, 1.0);
    const finalWmW = Math.round(wmW * scale);
    const finalWmH = Math.round(wmH * scale);
    const wmResized = await sharp(wmRaw)
      .resize(finalWmW, finalWmH)
      .ensureAlpha()
      .composite([{
        input: Buffer.from([0, 0, 0, Math.round(255 * WATERMARK_OPACITY)]),
        raw: { width: 1, height: 1, channels: 4 },
        tile: true,
        blend: "dest-in",
      }])
      .png()
      .toBuffer();
    const left = Math.max(0, imgW - finalWmW - WATERMARK_MARGIN);
    const top = Math.max(0, imgH - finalWmH - WATERMARK_MARGIN);
    return await sharp(imageBuffer)
      .composite([{ input: wmResized, left, top }])
      .jpeg({ quality: 92 })
      .toBuffer();
  } catch (e) {
    console.error("[render-final] Watermark failed (non-fatal):", e);
    return imageBuffer;
  }
}

/* ── Gemini helpers with retry ── */

async function geminiRetry<T>(fn: () => Promise<T>, maxAttempts = 3): Promise<T> {
  let last: unknown;
  for (let i = 1; i <= maxAttempts; i++) {
    try {
      return await fn();
    } catch (e) {
      last = e;
      const msg = String(e).toLowerCase();
      const transient = ["429", "500", "503", "overloaded", "deadline", "timeout", "unavailable", "internal"].some(k => msg.includes(k));
      if (!transient || i === maxAttempts) throw e;
      await new Promise(r => setTimeout(r, 2000 * i));
    }
  }
  throw last;
}

/* ── Masked composite (preserve non-wall areas from original) ── */

async function maskedComposite(
  originalBuf: Buffer,
  renderedBuf: Buffer,
  maskBuf: Buffer,
  W: number,
  H: number,
): Promise<Buffer> {
  try {
    const renderedResized = await sharp(renderedBuf).resize(W, H).jpeg({ quality: 95 }).toBuffer();
    const renderedWithAlpha = await sharp(renderedResized)
      .ensureAlpha()
      .composite([{ input: maskBuf, blend: "dest-in" }])
      .png()
      .toBuffer();
    const result = await sharp(originalBuf)
      .composite([{ input: renderedWithAlpha }])
      .jpeg({ quality: 92 })
      .toBuffer();
    return result;
  } catch (e) {
    console.warn("[render-final] maskedComposite failed (non-fatal):", e);
    return renderedBuf;
  }
}

/* ══════════════════════════════════════════════════════════
   POST handler — 3-phase pipeline
   ══════════════════════════════════════════════════════════ */

export async function POST(req: NextRequest) {
  const t0 = Date.now();
  const timings: Record<string, number> = {};

  const clientIp = getClientIp(req);
  const rateCheck = checkRateLimit(clientIp);
  if (!rateCheck.allowed) {
    return NextResponse.json(
      { error: `Dzienny limit generacji (${rateCheck.limit}) został wyczerpany. Spróbuj ponownie jutro.` },
      { status: 429 }
    );
  }

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

    /* ── 1. Decode and orient image ── */
    const imgBase64 = image.includes(",") ? image.split(",")[1] : image;
    const rawBuffer = Buffer.from(imgBase64, "base64");
    const rawMeta = await sharp(rawBuffer).metadata();
    const needsRotation = rawMeta.orientation !== undefined && rawMeta.orientation > 1;

    const imgBuffer = needsRotation
      ? await sharp(rawBuffer).rotate().jpeg({ quality: 95 }).toBuffer()
      : rawBuffer;
    const imgMeta = needsRotation ? await sharp(imgBuffer).metadata() : rawMeta;
    const origW = imgMeta.width!;
    const origH = imgMeta.height!;

    /* ── 2. Downscale for processing ── */
    const longest = Math.max(origW, origH);
    const downscale = longest > MAX_PROCESS_DIM ? MAX_PROCESS_DIM / longest : 1;
    const W = Math.round(origW * downscale);
    const H = Math.round(origH * downscale);

    const resizedImgBuffer = downscale < 1
      ? await sharp(imgBuffer).resize(W, H, { kernel: "lanczos3" }).jpeg({ quality: 92 }).toBuffer()
      : await sharp(imgBuffer).jpeg({ quality: 92 }).toBuffer();

    /* ── 3. Scale polygon to processed dimensions ── */
    const sx = W / canvas_width;
    const sy = H / canvas_height;
    const scaledPoly = polygon.map(p => ({ x: p.x * sx, y: p.y * sy }));

    /* ── 4. Compute polygon bounds ── */
    const polyXs = scaledPoly.map(p => p.x);
    const polyYs = scaledPoly.map(p => p.y);
    const polyWidth = Math.max(...polyXs) - Math.min(...polyXs);
    const polyHeight = Math.max(...polyYs) - Math.min(...polyYs);
    const polyHeightFraction = polyHeight / H;
    const polyWidthFraction = polyWidth / W;

    /* ── 5. Load texture + meta ── */
    const texturesDir = getTexturesDir();
    if (!texturesDir) {
      return NextResponse.json({ error: "Textures directory not found" }, { status: 500 });
    }
    const safeId = product_id.replace(/[^a-zA-Z0-9_-]/g, "");
    const texturePath = path.join(texturesDir, safeId, "albedo.jpg");
    const metaPath = path.join(texturesDir, safeId, "metadata.json");
    if (!fs.existsSync(texturePath) || !fs.existsSync(metaPath)) {
      return NextResponse.json({ error: `Product not found: ${safeId}` }, { status: 404 });
    }
    const textureFileBuffer = fs.readFileSync(texturePath);
    const meta = JSON.parse(fs.readFileSync(metaPath, "utf-8"));

    /* ── 6. Create mask overlay for dimension analysis ── */
    const points = scaledPoly.map(p => `${Math.round(p.x)},${Math.round(p.y)}`).join(" ");
    const overlaySvg = Buffer.from(
      `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">` +
      `<polygon points="${points}" fill="rgba(255,140,0,0.55)"/>` +
      `</svg>`
    );
    const overlayPng = await sharp(overlaySvg).png().toBuffer();
    const maskOverlayBuffer = await sharp(resizedImgBuffer)
      .composite([{ input: overlayPng }])
      .jpeg({ quality: 88 })
      .toBuffer();

    /* ── 7. PHASE 1: AI Scene Dimension Analysis ── */
    const t1 = Date.now();
    const apiKey = process.env.GEMINI_API_KEY;
    let sceneDims: SceneDimensions = {
      ...DEFAULT_DIMENSIONS,
      wallHeightCm: Math.round(270 * polyHeightFraction * 1.05),
      wallWidthCm: Math.round(270 * polyWidthFraction * (W / H) * 1.05),
    };

    if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey);
      sceneDims = await analyzeSceneDimensions(
        genAI, resizedImgBuffer, maskOverlayBuffer,
        polyHeightFraction, polyWidthFraction,
      );
      timings.dimension_analysis = Math.round((Date.now() - t1) / 100) / 10;

      /* ── 8. PHASE 2: Precise Texture Tiling ── */
      const t2 = Date.now();
      const texMeta = await sharp(textureFileBuffer).metadata();
      const scaleReport = computeTextureScale(
        polyHeight, polyWidth,
        texMeta.height!, texMeta.width!,
        meta, sceneDims,
      );
      console.log("[render-final] Scale report:", JSON.stringify(scaleReport));

      let tileW = Math.max(Math.round(texMeta.width! * scaleReport.scale), 1);
      let tileH = Math.max(Math.round(texMeta.height! * scaleReport.scale), 1);
      let tilesX = Math.ceil(W / tileW);
      let tilesY = Math.ceil(H / tileH);
      if (tilesX * tilesY > MAX_TILES) {
        const factor = Math.sqrt((tilesX * tilesY) / MAX_TILES);
        tileW = Math.round(tileW * factor);
        tileH = Math.round(tileH * factor);
        tilesX = Math.ceil(W / tileW);
        tilesY = Math.ceil(H / tileH);
      }

      const scaledTexBuf = await sharp(textureFileBuffer).resize(tileW, tileH).toBuffer();
      const tileOps: sharp.OverlayOptions[] = [];
      for (let y = 0; y < H; y += tileH) {
        for (let x = 0; x < W; x += tileW) {
          tileOps.push({ input: scaledTexBuf, left: x, top: y });
        }
      }
      const tiledBuffer = await sharp({
        create: { width: W, height: H, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 255 } },
      }).composite(tileOps).png().toBuffer();

      // Polygon mask
      const maskSvg = Buffer.from(
        `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">` +
        `<polygon points="${points}" fill="white"/>` +
        `</svg>`
      );
      const maskPng = await sharp(maskSvg).ensureAlpha().png().toBuffer();
      const grayMaskPng = await sharp(maskSvg).grayscale().png().toBuffer();

      const maskedTexture = await sharp(tiledBuffer)
        .composite([{ input: maskPng, blend: "dest-in" }])
        .png()
        .toBuffer();

      const compositeBuffer = await sharp(resizedImgBuffer)
        .composite([{ input: maskedTexture }])
        .jpeg({ quality: 92 })
        .toBuffer();

      const compositeB64 = "data:image/jpeg;base64," + compositeBuffer.toString("base64");
      timings.texture_project = Math.round((Date.now() - t2) / 100) / 10;

      /* ── 9. PHASE 3: Photorealistic Render (Nano Banana 2) ── */
      const t4 = Date.now();
      const imageModelId = process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image-preview";
      let geminiModel = imageModelId;
      let refinedB64 = compositeB64;

      try {
        const imageModel = genAI.getGenerativeModel({
          model: imageModelId,
          generationConfig: { temperature: 0.2 } as Record<string, unknown>,
        });
        const prompt = buildRenderPrompt(meta.name || product_id, meta, scaleReport, sceneDims);
        const result = await geminiRetry(() => imageModel.generateContent([
          { text: prompt },
          { inlineData: { mimeType: "image/jpeg", data: resizedImgBuffer.toString("base64") } },
          { inlineData: { mimeType: "image/jpeg", data: compositeBuffer.toString("base64") } },
          { inlineData: { mimeType: "image/jpeg", data: maskOverlayBuffer.toString("base64") } },
          { inlineData: { mimeType: "image/jpeg", data: textureFileBuffer.toString("base64") } },
        ]));

        const parts = result.response.candidates?.[0]?.content?.parts;
        const imagePart = parts?.find((p: { inlineData?: { mimeType: string; data: string } }) => p.inlineData);

        if (imagePart?.inlineData) {
          const renderedBuf = Buffer.from(imagePart.inlineData.data, "base64");
          const safeOutput = await maskedComposite(resizedImgBuffer, renderedBuf, grayMaskPng, W, H);
          refinedB64 = `data:image/jpeg;base64,${safeOutput.toString("base64")}`;
        } else {
          console.warn("[render-final] Image model returned no image part");
          geminiModel = `no-image: ${imageModelId}`;
        }
      } catch (e) {
        console.error("[render-final] Gemini render failed:", e);
        geminiModel = `error: ${e instanceof Error ? e.message : String(e)}`;
      }

      timings.gemini_render = Math.round((Date.now() - t4) / 100) / 10;

      /* ── 10. Watermark ── */
      const refinedRaw = refinedB64.includes(",") ? refinedB64.split(",")[1] : refinedB64;
      const refinedBuf = Buffer.from(refinedRaw, "base64");
      const wmBuf = await applyWatermark(refinedBuf);
      if (wmBuf !== refinedBuf) {
        refinedB64 = `data:image/jpeg;base64,${wmBuf.toString("base64")}`;
      }

      timings.total = Math.round((Date.now() - t0) / 100) / 10;

      recordUsage(clientIp, product_id, meta.name || product_id, geminiModel, timings);

      return NextResponse.json({
        composite: compositeB64,
        refined: refinedB64,
        gemini_model: geminiModel,
        timings,
        scale: {
          wallHeightCm: sceneDims.wallHeightCm,
          wallWidthCm: sceneDims.wallWidthCm,
          courses: scaleReport.expectedCoursesH,
          unitsPerRow: scaleReport.expectedUnitsW,
          dimensionSource: scaleReport.dimensionSource,
          sceneType: sceneDims.sceneType,
          referenceObjects: sceneDims.referenceObjects.map(r => r.name),
        },
      });
    }

    // No API key — return composite only
    const texMeta = await sharp(textureFileBuffer).metadata();
    const fallbackReport = computeTextureScale(
      polyHeight, polyWidth,
      texMeta.height!, texMeta.width!,
      meta, sceneDims,
    );
    let tileW = Math.max(Math.round(texMeta.width! * fallbackReport.scale), 1);
    let tileH = Math.max(Math.round(texMeta.height! * fallbackReport.scale), 1);
    let tilesX = Math.ceil(W / tileW);
    let tilesY = Math.ceil(H / tileH);
    if (tilesX * tilesY > MAX_TILES) {
      const factor = Math.sqrt((tilesX * tilesY) / MAX_TILES);
      tileW = Math.round(tileW * factor);
      tileH = Math.round(tileH * factor);
      tilesX = Math.ceil(W / tileW);
      tilesY = Math.ceil(H / tileH);
    }
    const scaledTexBuf = await sharp(textureFileBuffer).resize(tileW, tileH).toBuffer();
    const tileOps: sharp.OverlayOptions[] = [];
    for (let y = 0; y < H; y += tileH) {
      for (let x = 0; x < W; x += tileW) {
        tileOps.push({ input: scaledTexBuf, left: x, top: y });
      }
    }
    const tiledBuffer = await sharp({
      create: { width: W, height: H, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 255 } },
    }).composite(tileOps).png().toBuffer();
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
    const compositeB64 = "data:image/jpeg;base64," + compositeBuffer.toString("base64");

    timings.total = Math.round((Date.now() - t0) / 100) / 10;
    return NextResponse.json({ composite: compositeB64, refined: compositeB64, gemini_model: "not-configured", timings });
  } catch (e) {
    console.error("[render-final]", e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Render failed" },
      { status: 500 }
    );
  }
}
