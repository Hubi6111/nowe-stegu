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
  const moduleH = scaleReport.brickBodyHeightMm;
  const moduleW = scaleReport.brickBodyWidthMm;
  const joint = scaleReport.jointMm;
  const layout = String(meta.layoutType || "running-bond");
  const isBrick = layout === "running-bond" || layout === "stretcher-bond";
  const materialType = isBrick ? "decorative brick cladding" : "decorative wall panel";

  const courseH = moduleH + joint;
  const bricksNextToDoor = Math.round(2000 / courseH);
  const bricksFullWall = Math.round((sceneDims.ceilingHeightCm || 270) * 10 / courseH);

  const refObjectsDesc = sceneDims.referenceObjects.length > 0
    ? sceneDims.referenceObjects.map(r => `  • ${r.name}: ~${r.estimatedSizeCm} cm (${r.confidence} confidence)`).join("\n")
    : "  • No specific reference objects detected — using standard room proportions";

  const dimBlock = isBrick
    ? `PRODUCT: ${materialType} — "${productName}"
  • Single brick body: ${moduleW} mm wide × ${moduleH} mm tall (${(moduleW / 10).toFixed(1)} × ${(moduleH / 10).toFixed(1)} cm)
  • Mortar joint thickness: ${joint} mm (${(joint / 10).toFixed(1)} cm)
  • One course height (brick + mortar): ${courseH} mm (${(courseH / 10).toFixed(1)} cm)
  • Brick aspect ratio: ${(moduleW / moduleH).toFixed(1)}:1 (each brick is ${(moduleW / moduleH).toFixed(1)}× wider than tall)`
    : `PRODUCT: ${materialType} — "${productName}"
  • Single panel: ${moduleW} mm wide × ${moduleH} mm tall (${(moduleW / 10).toFixed(1)} × ${(moduleH / 10).toFixed(1)} cm)
  • Gap between panels: ${joint} mm`;

  return `You are an expert photorealistic interior/exterior rendering engine. Your #1 priority: PHYSICALLY CORRECT BRICK/TILE DIMENSIONS. The textured wall must be indistinguishable from a real renovation photograph.

YOU RECEIVE FOUR IMAGES (in order):
  1. ORIGINAL — the unmodified room/building photo
  2. COMPOSITE — room with texture algorithmically placed at calculated real-world scale
  3. MASK OVERLAY — ORIGINAL with ORANGE highlight = user's wall selection
  4. PRODUCT TEXTURE TILE — the real decorative material to apply

═══ WALL DIMENSIONS (AI-estimated from reference objects) ═══
Scene type: ${sceneDims.sceneType}
Highlighted wall area: ~${sceneDims.wallHeightCm} cm tall × ~${sceneDims.wallWidthCm} cm wide
${sceneDims.ceilingHeightCm ? `Ceiling height: ~${sceneDims.ceilingHeightCm} cm` : ""}
Dimension confidence: ${sceneDims.confidence}
Reference objects used:
${refObjectsDesc}

═══ PRODUCT PHYSICAL DIMENSIONS ═══
${dimBlock}

═══ CRITICAL SCALE REQUIREMENTS ═══
${isBrick ? `At these dimensions, the wall MUST contain:
  • VERTICALLY: ~${scaleReport.expectedCoursesH} brick courses (each ${(courseH / 10).toFixed(1)} cm tall)
  • HORIZONTALLY: ~${scaleReport.expectedUnitsW} bricks per row (each ${(moduleW / 10).toFixed(1)} cm wide + ${(joint / 10).toFixed(1)} cm mortar)

DIMENSIONAL ANCHORS — verify your output against these:
  • A standard door (200 cm) should have ~${bricksNextToDoor} brick courses beside it
  • Floor-to-ceiling (~${sceneDims.ceilingHeightCm || 270} cm) should have ~${bricksFullWall} courses total
  • A light switch (at ~115 cm from floor) should be at approximately course ${Math.round(1150 / courseH)}
  • Each individual brick should appear ${(moduleH / 10).toFixed(1)} cm tall — about the width of an adult's palm
  • 10 stacked bricks (with mortar) = ${(courseH * 10 / 10).toFixed(0)} cm ≈ knee height

The COMPOSITE image (image 2) already has ~${scaleReport.coursesInPoly} courses and ~${scaleReport.unitsInPolyRow} bricks per row at the correct scale.
YOU MUST MATCH THIS EXACT SCALE. Count the courses in the COMPOSITE and reproduce the same count.` :
`At these dimensions, the wall should contain:
  • VERTICALLY: ~${scaleReport.expectedCoursesH} panel rows
  • HORIZONTALLY: ~${scaleReport.expectedUnitsW} panels per row

The COMPOSITE image already has the correct scale — match it exactly.`}

═══ PERSPECTIVE & GEOMETRY ═══
  • Study the ORIGINAL photo for vanishing points, camera tilt, and lens distortion
  • Perspective angle: ${sceneDims.perspectiveAngle}
  • The texture MUST follow the same perspective foreshortening as the wall surface
  • Bricks further from camera appear smaller — this is correct perspective, not a scale error
  • Mortar lines must converge toward vanishing points like any real surface
  • The COMPOSITE already has correct flat tiling — add perspective warping to match the wall plane

═══ PHOTOREALISTIC RENDERING ═══
  First, analyze the ORIGINAL image lighting:
  • Identify the dominant light direction (left, right, top, diffuse)
  • Note the color temperature (warm, cool, neutral)
  • Observe shadow intensity on existing objects

  Then apply:
  • Ambient occlusion at ceiling/floor/corner junctions — subtle darkening
  • Contact shadows where furniture meets the textured wall
  • Surface relief: ${isBrick ? "each brick must show individual 3D depth — highlight on top edge, shadow on bottom edge, recessed mortar joints with micro-shadows" : "each panel with subtle edge shadows and visible gaps between panels"}
  • Natural edge blending where texture meets non-textured areas
  • Match the room's color temperature, exposure, white balance, and noise/grain
  • Texture color must match image 4 (PRODUCT TEXTURE) — preserve the exact hue and tone
  • Everything outside the textured area = pixel-perfect identical to ORIGINAL

═══ ABSOLUTE RESTRICTIONS ═══
  • Do NOT zoom, crop, pan, rotate, or change image dimensions
  • Do NOT extend texture beyond the orange mask boundary
  • Do NOT texture ceiling, floor, or other walls
  • Do NOT cover or alter any furniture, objects, windows, doors, electrical fixtures
  • Do NOT change the brick/tile count — KEEP the same scale as the COMPOSITE
  • Do NOT stretch or squash bricks — maintain the ${(moduleW / moduleH).toFixed(1)}:1 aspect ratio

Return ONLY the final photorealistic image. No text, no explanation.`;
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

      /* ── 9. PHASE 3: Gemini Photorealistic Render ── */
      const t4 = Date.now();
      const imageModelId = process.env.GEMINI_IMAGE_MODEL || "gemini-2.0-flash-exp";
      let geminiModel = imageModelId;
      let refinedB64 = compositeB64;

      const prompt = buildRenderPrompt(meta.name || product_id, meta, scaleReport, sceneDims);
      const inputParts = [
        { text: prompt },
        { inlineData: { mimeType: "image/jpeg", data: resizedImgBuffer.toString("base64") } },
        { inlineData: { mimeType: "image/jpeg", data: compositeBuffer.toString("base64") } },
        { inlineData: { mimeType: "image/jpeg", data: maskOverlayBuffer.toString("base64") } },
        { inlineData: { mimeType: "image/jpeg", data: textureFileBuffer.toString("base64") } },
      ];

      // Try with responseModalities first (needed for gemini-2.0-flash-exp),
      // fall back to plain config (works for gemini-*-image-preview models)
      for (const attempt of [1, 2] as const) {
        try {
          const genConfig = attempt === 1
            ? { responseModalities: ["Text", "Image"], temperature: 0.2 }
            : { temperature: 0.2 };
          const imageModel = genAI.getGenerativeModel({
            model: imageModelId,
            generationConfig: genConfig as Record<string, unknown>,
          });
          const result = await geminiRetry(() => imageModel.generateContent(inputParts), 2);

          const parts = result.response.candidates?.[0]?.content?.parts;
          const imagePart = parts?.find((p: { inlineData?: { mimeType: string; data: string } }) => p.inlineData);

          if (imagePart?.inlineData) {
            const renderedBuf = Buffer.from(imagePart.inlineData.data, "base64");
            const safeOutput = await maskedComposite(resizedImgBuffer, renderedBuf, grayMaskPng, W, H);
            refinedB64 = `data:image/jpeg;base64,${safeOutput.toString("base64")}`;
            break;
          }

          if (attempt === 1) {
            console.warn(`[render-final] Attempt ${attempt}: no image returned, retrying without responseModalities`);
            continue;
          }
          console.warn("[render-final] Gemini returned no image in either attempt");
          geminiModel = `no-image: ${imageModelId}`;
        } catch (e) {
          if (attempt === 1) {
            console.warn(`[render-final] Attempt ${attempt} failed, trying fallback config:`, e instanceof Error ? e.message : String(e));
            continue;
          }
          console.error("[render-final] Gemini render failed:", e);
          geminiModel = `error: ${e instanceof Error ? e.message : String(e)}`;
        }
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
