import { NextRequest, NextResponse } from "next/server";
import sharp from "sharp";
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { getClientIp, checkRateLimit, recordUsage } from "../rate-limit";

export const maxDuration = 300;

interface Point { x: number; y: number }

const WALL_HEIGHT_MM = 2850;
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

/* ── Texture scale computation with dimensional verification ── */

interface ScaleReport {
  scale: number;
  wallHeightMm: number;
  wallWidthMm: number;
  expectedCoursesH: number;
  expectedUnitsW: number;
  courseHeightMm: number;
  unitWidthMm: number;
  coursesInPoly: number;
  unitsInPolyRow: number;
  verified: boolean;
}

function computeTextureScale(
  polyHeightPx: number,
  polyWidthPx: number,
  imageHeightPx: number,
  imageWidthPx: number,
  textureHeightPx: number,
  textureWidthPx: number,
  meta: Record<string, unknown>
): ScaleReport {
  const moduleH = Number(meta.moduleHeightMm || 65);
  const moduleW = Number(meta.moduleWidthMm || 245);
  const joint = Number(meta.jointMm || 10);
  const layout = String(meta.layoutType || "running-bond");
  let courses = Number(meta.albedoBrickCourses || meta.albedoCourses || 8);
  courses = Math.max(1, Math.min(courses, 32));
  const isBrick = layout === "running-bond" || layout === "stretcher-bond";

  const courseH = moduleH + joint;
  const unitW = moduleW + joint;

  // Physical size of one albedo tile
  let albedoHMm: number;
  let albedoWMm: number;
  if (isBrick) {
    albedoHMm = moduleH * courses + joint * Math.max(courses - 1, 0);
    const bricksPerRow = Math.max(1, Math.round(textureWidthPx / textureHeightPx * courses));
    albedoWMm = moduleW * bricksPerRow + joint * Math.max(bricksPerRow - 1, 0);
  } else {
    const planks = Math.max(1, Math.min(Number(meta.albedoStackPlanks || meta.albedoPlankCount || 2), 12));
    albedoHMm = moduleH * planks + joint * Math.max(planks - 1, 0);
    albedoWMm = moduleW;
  }

  // Estimate real-world wall height from polygon coverage fraction
  const fracH = polyHeightPx / imageHeightPx;

  let wallHMm: number;
  if (fracH >= 0.6) {
    wallHMm = WALL_HEIGHT_MM;
  } else if (fracH >= 0.35) {
    wallHMm = WALL_HEIGHT_MM * (0.7 + 0.3 * ((fracH - 0.35) / 0.25));
  } else {
    wallHMm = Math.max(600, WALL_HEIGHT_MM * fracH * 1.8);
  }

  // Derive wall width from aspect ratio
  let wallWMm = wallHMm * (polyWidthPx / polyHeightPx);

  // Cross-validate: typical interior wall 1000-6000mm wide
  if (wallWMm > 8000) {
    const correction = 6000 / wallWMm;
    wallHMm *= correction;
    wallWMm = wallHMm * (polyWidthPx / polyHeightPx);
  } else if (wallWMm < 600 && polyWidthPx / imageWidthPx > 0.2) {
    wallHMm *= 1.2;
    wallWMm = wallHMm * (polyWidthPx / polyHeightPx);
  }

  // How many courses and units should fit
  const expectedCoursesH = wallHMm / courseH;
  const expectedUnitsW = wallWMm / unitW;

  // Target pixel sizes for one albedo tile
  const pxPerMm = polyHeightPx / wallHMm;
  const targetAlbedoHPx = albedoHMm * pxPerMm;
  const scaleFromH = targetAlbedoHPx / textureHeightPx;

  // Verify: count how many courses would actually fit in polygon at this scale
  const tileHPx = textureHeightPx * scaleFromH;
  const coursesInPoly = (polyHeightPx / tileHPx) * courses;

  // Verify width
  const tileWPx = textureWidthPx * scaleFromH;
  const albedoRepeatsW = polyWidthPx / tileWPx;
  const bricksPerAlbedo = isBrick ? Math.max(1, Math.round(textureWidthPx / textureHeightPx * courses)) : 1;
  const unitsInPolyRow = albedoRepeatsW * bricksPerAlbedo;

  // Compare expected vs actual — if off by >30%, adjust
  const hRatio = coursesInPoly / expectedCoursesH;
  let finalScale = scaleFromH;
  let verified = true;

  if (hRatio < 0.7 || hRatio > 1.3) {
    // Recalculate to match expected course count exactly
    const targetTileH = (polyHeightPx / expectedCoursesH) * courses;
    finalScale = targetTileH / textureHeightPx;
    verified = false;
  }

  const mult = Math.max(0.35, Math.min(Number(meta.textureScaleMultiplier || 1), 2.5));
  finalScale = Math.max(0.02, finalScale * mult);

  return {
    scale: finalScale,
    wallHeightMm: Math.round(wallHMm),
    wallWidthMm: Math.round(wallWMm),
    expectedCoursesH: Math.round(expectedCoursesH),
    expectedUnitsW: Math.round(expectedUnitsW),
    courseHeightMm: courseH,
    unitWidthMm: unitW,
    coursesInPoly: Math.round(coursesInPoly),
    unitsInPolyRow: Math.round(unitsInPolyRow),
    verified,
  };
}

/* ── Scene analysis prompt (mirrors Python analyze_wall_scene) ── */

const SCENE_ANALYSIS_PROMPT = `You are analysing a room photo where a wall texture has been applied.

You receive THREE images:
  1. ORIGINAL room photo
  2. COMPOSITE — room with texture applied on wall at algorithmically computed real-world scale
  3. MASK OVERLAY — orange highlight showing where texture is placed

Analyse the scene and OUTPUT a single JSON object (no markdown, no code fences):
{"lighting_direction":"left","lighting_temperature":"warm","ambient_level":0.65,"shadow_intensity":"medium","blend_notes":"soft fade at ceiling; crisp edge at shelf","texture_scale_looks_correct":true,"scale_notes":"bricks appear correctly sized relative to furniture"}

Fields:
- lighting_direction: "left", "right", "top", "diffuse"
- lighting_temperature: "warm", "cool", "neutral"
- ambient_level: 0.0–1.0
- shadow_intensity: "light", "medium", "strong"
- blend_notes: brief note about how texture edges should blend at boundaries
- texture_scale_looks_correct: true/false — does the brick/tile size in image 2 look physically realistic compared to objects in image 1?
- scale_notes: brief note about whether bricks/tiles appear correctly sized relative to doors, furniture, power sockets, etc.`;

/* ── Full photorealistic render prompt (mirrors Python _RENDER_PROMPT_TEMPLATE) ── */

function buildRenderPrompt(
  productName: string,
  meta: Record<string, unknown>,
  analysis: Record<string, unknown>,
  scaleReport: ScaleReport
): string {
  const moduleH = Number(meta.moduleHeightMm || 80);
  const moduleW = Number(meta.moduleWidthMm || 245);
  const joint = Number(meta.jointMm || 10);
  const layout = String(meta.layoutType || "running-bond");
  const isBrick = layout === "running-bond" || layout === "stretcher-bond";
  const materialType = isBrick ? "decorative brick cladding" : "decorative wall panel";

  const dimInstructions = isBrick
    ? `Product: ${materialType} — "${productName}".
  • Each brick: ${moduleW} mm wide × ${moduleH} mm tall (≈ ${(moduleW / 10).toFixed(1)} × ${(moduleH / 10).toFixed(1)} cm)
  • Mortar joint: ${joint} mm (≈ ${(joint / 10).toFixed(1)} cm)
  • Course height (brick + joint): ${moduleH + joint} mm (≈ ${((moduleH + joint) / 10).toFixed(1)} cm)`
    : `Product: ${materialType} — "${productName}".
  • Each panel: ${moduleW} mm wide × ${moduleH} mm tall
  • Gap between panels: ${joint} mm`;

  const scaleVerification = `
═══ ALGORITHMIC SCALE VERIFICATION (pre-computed) ═══
Our algorithm estimated the selected wall area to be approximately:
  • Wall height: ~${scaleReport.wallHeightMm} mm (${(scaleReport.wallHeightMm / 10).toFixed(0)} cm)
  • Wall width: ~${scaleReport.wallWidthMm} mm (${(scaleReport.wallWidthMm / 10).toFixed(0)} cm)

At correct real-world scale, this wall should contain:
  • ~${scaleReport.expectedCoursesH} ${isBrick ? "brick courses" : "panel rows"} vertically (each ${scaleReport.courseHeightMm} mm tall)
  • ~${scaleReport.expectedUnitsW} ${isBrick ? "bricks" : "panels"} per horizontal row (each ${scaleReport.unitWidthMm} mm wide)

The COMPOSITE image (image 2) was tiled with these parameters. It contains ~${scaleReport.coursesInPoly} courses and ~${scaleReport.unitsInPolyRow} units per row.
${scaleReport.verified ? "✓ Scale verified — composite matches expected dimensions." : "⚠ Scale was auto-corrected — the composite may have minor deviations."}

CRITICAL: You MUST preserve this exact scale in your render. Count the ${isBrick ? "brick courses" : "rows"} in the COMPOSITE and ensure your output has the same number. If you see ~${scaleReport.coursesInPoly} courses in the composite, your output must also show ~${scaleReport.coursesInPoly} courses.`;

  const lightDir = String(analysis.lighting_direction || "diffuse");
  const lightTemp = String(analysis.lighting_temperature || "neutral");
  const shadowInt = String(analysis.shadow_intensity || "medium");
  const blendNotes = String(analysis.blend_notes || "natural fade at all edges");

  return `You are an expert photorealistic interior rendering engine. Your goal: make the textured wall look indistinguishable from a real renovation photograph.

YOU RECEIVE FOUR IMAGES (in order):
  1. ORIGINAL — the unmodified room photo
  2. COMPOSITE — room with texture algorithmically placed at verified real-world scale
  3. MASK OVERLAY — ORIGINAL with ORANGE highlight = user's selection area
  4. PRODUCT TEXTURE TILE — the real decorative material

═══ SCENE ANALYSIS ═══
  1. In the ORIGINAL, identify: wall surface vs non-wall (furniture, objects, ceiling, floor, other walls)
  2. Apply texture ONLY on flat wall surface within the orange selection
  3. Keep everything else from the ORIGINAL untouched (furniture, objects, etc.)

═══ PHYSICAL DIMENSIONS ═══
${dimInstructions}
${scaleVerification}

Use reference objects to double-check: standard doors ~200 cm, light switches ~120 cm from floor, ceiling ~270-285 cm.
A standard doorway should fit ~${Math.round(2000 / (moduleH + joint))} courses next to it.

═══ PERSPECTIVE & SCALE ═══
  • Study the ORIGINAL for vanishing points and camera angle
  • The texture must follow the same perspective foreshortening
  • The COMPOSITE already has the correct scale — match it exactly
  • Every ${isBrick ? "brick" : "panel"} must have identical physical size (only perspective foreshortening allowed)

═══ PHOTOREALISTIC RENDERING ═══
  • Lighting: ${lightTemp} from ${lightDir}
  • Ambient occlusion at ceiling/floor junctions
  • Contact shadows where furniture meets wall (intensity: ${shadowInt})
  • Surface relief: individual ${isBrick ? "bricks with edge shadows, recessed mortar lines" : "panels with subtle edge shadows, visible gaps"}
  • Edge blending: ${blendNotes}
  • Match room's color temperature, exposure, white balance
  • Everything outside textured area = identical to ORIGINAL

═══ RESTRICTIONS ═══
  • Do NOT zoom, crop, pan, or change dimensions
  • Do NOT extend texture beyond orange boundary
  • Do NOT texture ceiling, floor, or other walls
  • Do NOT cover furniture or objects
  • KEEP the same tiling pattern and scale as the COMPOSITE

Return ONLY the final photorealistic image. No text.`;
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

/* ── Scene analysis (Step 1) ── */

async function analyzeWallScene(
  genAI: GoogleGenerativeAI,
  originalBuf: Buffer,
  compositeBuf: Buffer,
  maskOverlayBuf: Buffer,
): Promise<Record<string, unknown>> {
  const defaults = {
    lighting_direction: "diffuse",
    lighting_temperature: "neutral",
    ambient_level: 0.6,
    shadow_intensity: "medium",
    blend_notes: "natural fade at all edges",
  };
  const textModelId = process.env.GEMINI_TEXT_MODEL || "gemini-2.5-pro";
  try {
    const model = genAI.getGenerativeModel({ model: textModelId });
    const result = await geminiRetry(() => model.generateContent([
      { text: SCENE_ANALYSIS_PROMPT },
      { inlineData: { mimeType: "image/jpeg", data: originalBuf.toString("base64") } },
      { inlineData: { mimeType: "image/jpeg", data: compositeBuf.toString("base64") } },
      { inlineData: { mimeType: "image/jpeg", data: maskOverlayBuf.toString("base64") } },
    ]));
    const text = result.response.text?.() ?? "";
    // strip markdown fences if present
    const clean = text.replace(/```(?:json)?\s*/g, "").replace(/```/g, "").trim();
    const m = clean.match(/\{[\s\S]*\}/);
    if (m) {
      try {
        const parsed = JSON.parse(m[0]);
        return { ...defaults, ...parsed };
      } catch { /* fall through */ }
    }
  } catch (e) {
    console.warn("[render-final] Scene analysis failed (non-fatal):", e);
  }
  return defaults;
}

/* ── Masked composite (Python safety-net: preserve non-wall areas from original) ── */

async function maskedComposite(
  originalBuf: Buffer,
  renderedBuf: Buffer,
  maskBuf: Buffer, // grayscale mask PNG
  W: number,
  H: number,
): Promise<Buffer> {
  try {
    // Resize rendered image to match original dimensions (Gemini may return slightly different size)
    const renderedResized = await sharp(renderedBuf).resize(W, H).jpeg({ quality: 95 }).toBuffer();

    // Create RGBA rendered image with mask as alpha channel
    // Where mask is white (255) → use rendered; where black (0) → use original
    const renderedWithAlpha = await sharp(renderedResized)
      .ensureAlpha()
      .composite([{ input: maskBuf, blend: "dest-in" }])
      .png()
      .toBuffer();

    // Composite: original underneath, rendered with mask alpha on top
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
   POST handler
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

    /* ── 4. Load texture + meta ── */
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

    /* ── 5. Texture tiling ── */
    const t1 = Date.now();
    const texMeta = await sharp(textureFileBuffer).metadata();
    const polyXs = scaledPoly.map(p => p.x);
    const polyYs = scaledPoly.map(p => p.y);
    const polyWidth = Math.max(...polyXs) - Math.min(...polyXs);
    const polyHeight = Math.max(...polyYs) - Math.min(...polyYs);
    const scaleReport = computeTextureScale(polyHeight, polyWidth, H, W, texMeta.height!, texMeta.width!, meta);
    const texScale = scaleReport.scale;
    console.log("[render-final] Scale report:", JSON.stringify(scaleReport));

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

    /* ── 6. Polygon mask ── */
    const points = scaledPoly.map(p => `${Math.round(p.x)},${Math.round(p.y)}`).join(" ");
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
    timings.texture_project = Math.round((Date.now() - t1) / 100) / 10;

    /* ── 7. Mask overlay (orange highlight for Gemini) ── */
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

    /* ── 8. Two-step Gemini pipeline ── */
    const t2 = Date.now();
    const apiKey = process.env.GEMINI_API_KEY;
    let refinedB64 = compositeB64;
    let geminiModel = "not-configured";

    if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey);

      // Step 1: Scene analysis (text model)
      const analysis = await analyzeWallScene(genAI, resizedImgBuffer, compositeBuffer, maskOverlayBuffer);

      // Step 2: Photorealistic render (image model)
      const imageModelId = process.env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image-preview";
      geminiModel = imageModelId;
      try {
        const imageModel = genAI.getGenerativeModel({
          model: imageModelId,
          generationConfig: { temperature: 0.2 } as Record<string, unknown>,
        });
        const prompt = buildRenderPrompt(meta.name || product_id, meta, analysis, scaleReport);
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

          // Safety net: clip Gemini output to mask (preserve non-wall areas from original)
          const safeOutput = await maskedComposite(resizedImgBuffer, renderedBuf, grayMaskPng, W, H);
          refinedB64 = `data:image/jpeg;base64,${safeOutput.toString("base64")}`;
        }
      } catch (e) {
        console.error("[render-final] Gemini render failed:", e);
        geminiModel = `error: ${e instanceof Error ? e.message : String(e)}`;
      }
    }

    timings.gemini_render = Math.round((Date.now() - t2) / 100) / 10;

    /* ── 9. Watermark ── */
    const refinedRaw = refinedB64.includes(",") ? refinedB64.split(",")[1] : refinedB64;
    const refinedBuf = Buffer.from(refinedRaw, "base64");
    const wmBuf = await applyWatermark(refinedBuf);
    if (wmBuf !== refinedBuf) {
      refinedB64 = `data:image/jpeg;base64,${wmBuf.toString("base64")}`;
    }

    timings.total = Math.round((Date.now() - t0) / 100) / 10;

    recordUsage(clientIp, product_id, meta.name || product_id, geminiModel, timings);

    return NextResponse.json({ composite: compositeB64, refined: refinedB64, gemini_model: geminiModel, timings });
  } catch (e) {
    console.error("[render-final]", e);
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Render failed" },
      { status: 500 }
    );
  }
}
