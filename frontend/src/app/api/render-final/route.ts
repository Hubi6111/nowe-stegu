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

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/* ══════════════════════════════════════════════════════════
   STEP 1: Gemini 2.5 Pro — Scene Analysis
   Measures wall dimensions to the centimeter using every
   available reference object, analyzes lighting, perspective
   ══════════════════════════════════════════════════════════ */

const SCENE_ANALYSIS_PROMPT = `You are a forensic architectural measurement system. Your task: measure the EXACT real-world dimensions of a wall area and catalog everything visible in the scene for photorealistic rendering.

You receive THREE images:
  1. ORIGINAL — unmodified room/building photograph
  2. COMPOSITE — same photo with decorative texture tiled on the wall
  3. MASK OVERLAY — ORIGINAL with ORANGE semi-transparent highlight = target wall area

═══ TASK A: MULTI-REFERENCE DIMENSIONAL CALIBRATION ═══

You MUST find and use EVERY reference object visible anywhere in the photo to triangulate the wall dimensions. Known real-world sizes (use as many as apply):

DOORS (PRIMARY — most reliable, error ±3%):
  • Standard interior door leaf: 200 cm × 80 cm
  • Door frame/architrave adds: 6-8 cm each side, 5-8 cm top
  • Total opening with frame: ~210 cm × ~95 cm
  • French/balcony door: 200-220 cm × 80-180 cm
  • Closet sliding door: 230-250 cm × 60-90 cm per panel
  • Front/exterior door: 200-210 cm × 90-100 cm

WINDOWS (error ±5%):
  • Standard window: 120-150 cm tall × 80-120 cm wide
  • Window sill from floor: 85-95 cm (interior), 90-100 cm (exterior)
  • Curtain rod to ceiling: 10-20 cm
  • Roller blind cassette: 7-10 cm tall

ARCHITECTURAL (error ±3%):
  • Standard EU ceiling height: 250-280 cm (most common: 260-270 cm)
  • Standard US ceiling height: 244 cm (8 ft) or 274 cm (9 ft)
  • Baseboard/skirting: 8-12 cm tall
  • Crown molding: 5-10 cm
  • Stair riser: 17-19 cm, tread: 25-30 cm
  • Standard brick (visible outside): 6.5 cm × 25 cm
  • Concrete block: 19 cm × 39 cm
  • Radiator height: 30 cm (low), 50-60 cm (standard), 90+ cm (tall)
  • Radiator from floor: 10-15 cm

ELECTRICAL (error ±5%):
  • Light switch plate: 8 × 8 cm, center 110-120 cm from floor
  • Double switch plate: 8 × 15 cm
  • Power outlet plate: 8 × 8 cm, center 25-30 cm from floor
  • Kitchen counter outlet: 100-110 cm from floor
  • Thermostat: 8 × 12 cm, center ~150 cm from floor

FURNITURE (error ±8%):
  • Dining table height: 73-76 cm
  • Kitchen counter/island: 85-90 cm tall
  • Bar counter: 100-110 cm tall
  • Chair seat: 43-47 cm from floor
  • Chair backrest top: 85-100 cm from floor
  • Sofa seat: 40-45 cm, backrest: 80-90 cm total
  • Coffee table: 40-50 cm tall
  • Bookshelf: 180-200 cm (standard), 70-80 cm (low)
  • Desk: 72-76 cm tall
  • Bed frame headboard: 90-120 cm from floor
  • Bedside table: 55-65 cm tall

APPLIANCES/OBJECTS (error ±3%):
  • TV 55": ~68 × 122 cm. TV 65": ~81 × 144 cm
  • Refrigerator: 170-185 cm tall (standard), 60-70 cm wide
  • Washing machine: 85 cm tall × 60 cm wide
  • Microwave: 28-30 cm tall
  • Standard A4 frame: 21 × 30 cm
  • Wall clock: 25-40 cm diameter
  • Wine bottle on shelf: 30 cm tall
  • Standard door handle from floor: 100-110 cm

HUMAN PROPORTIONS (if people visible, error ±5%):
  • Adult standing: 165-180 cm
  • Shoulder height: 140-155 cm
  • Elbow height: 100-110 cm

MEASUREMENT PROCEDURE (execute ALL steps):
  1. SCAN the entire image systematically — left to right, top to bottom
  2. LIST every reference object you can identify with its approximate pixel span
  3. For the 3 most reliable references, calculate px/cm ratio independently
  4. CROSS-CHECK: all ratios should agree within 15%. Use the median.
  5. Apply the calibrated px/cm to the orange-highlighted wall area
  6. VERIFY: does the resulting wall height make sense given ceiling height?
  7. VERIFY: would a standard door (200cm) fit proportionally?

═══ TASK B: OCCLUSION CATALOG ═══

List EVERY object that is IN FRONT OF or ON the wall surface within the orange-highlighted area. These objects must NOT be covered by texture:
  • Furniture touching or near the wall (shelves, TV, frames, mirrors)
  • Architectural elements (columns, beams, window frames, door frames)
  • Electrical (switches, outlets, thermostats)
  • Decorative (clocks, plants, lamps, sconces, artwork)
  • Structural (pipes, vents, radiators)

For each occluder, provide its bounding box in normalized 0-1 coordinates relative to the full image.

═══ TASK C: WALL BOUNDARY ANALYSIS ═══

The orange mask is the USER's rough selection. But texture must ONLY go on the actual wall surface. Identify precisely:
  • Where does the wall meet the CEILING? (y coordinate, normalized 0-1)
  • Where does the wall meet the FLOOR/baseboard? (y coordinate, normalized 0-1)
  • Where are LEFT and RIGHT wall boundaries? (corners, door frames, windows)
  • Does the orange selection extend BEYOND the wall onto ceiling/floor/other walls?
  • If yes: which parts of the selection should be EXCLUDED?

═══ TASK D: LIGHTING ANALYSIS ═══

Analyze the EXACT lighting on the wall surface:
  • Primary light source position relative to wall
  • Secondary/fill light sources
  • Color temperature: warm (2700-3500K), neutral (4000-5000K), cool (5500K+)
  • Brightness gradient across the wall (which region is brightest/darkest?)
  • Shadow characteristics: hard (direct sun) vs soft (overcast/diffuse)
  • Ambient occlusion darkness at ceiling/floor/corner junctions
  • Any color cast from nearby colored surfaces
  • Overall exposure level

═══ TASK E: PERSPECTIVE ═══

  • Camera angle to wall: frontal (0-10°), moderate (10-30°), strong (30°+)
  • Which side recedes? (left/right)
  • Vanishing point direction for horizontal lines
  • Lens distortion visible?

═══ TASK F: TEXTURE SCALE CHECK ═══

Look at image 2 (COMPOSITE). Do the texture elements appear the correct physical size compared to reference objects? Are they too big or too small?

OUTPUT a single JSON object (no markdown, no backticks, no explanation):
{"wallHeightCm":255,"wallWidthCm":340,"ceilingHeightCm":265,"measurementMethod":"door frame at left edge used as primary (200cm), light switch confirmed (115cm from floor), ceiling height cross-checked","referenceObjects":[{"name":"interior door","pixelHeight":520,"realHeightCm":200,"pxPerCm":2.6,"confidence":"high"},{"name":"light switch","pixelHeight":21,"realHeightCm":8,"pxPerCm":2.63,"confidence":"medium"},{"name":"baseboard","pixelHeight":26,"realHeightCm":10,"pxPerCm":2.6,"confidence":"medium"}],"calibratedPxPerCm":2.6,"occluders":[{"x":0.05,"y":0.3,"w":0.15,"h":0.65,"label":"bookshelf","depth":"touching_wall"},{"x":0.7,"y":0.5,"w":0.08,"h":0.04,"label":"light_switch","depth":"on_wall"}],"wallBoundaries":{"ceilingLineY":0.02,"floorLineY":0.95,"leftEdgeX":0.0,"rightEdgeX":1.0,"selectionExceedsCeiling":false,"selectionExceedsFloor":false,"selectionExceedsLeftWall":false,"selectionExceedsRightWall":false},"lighting":{"primarySource":"window left, natural daylight","secondarySource":"ceiling fixture, warm LED","temperature":"neutral-warm","temperatureKelvin":4500,"gradient":"brighter-left-darker-right","gradientIntensity":0.3,"shadowType":"soft-diffuse","shadowIntensity":0.4,"ambientOcclusion":"visible at ceiling junction","colorCast":"slight warm cast from floor","exposure":"correct"},"perspective":{"type":"frontal","angleDeg":5,"recedes":"none","horizontalConvergence":"negligible"},"textureScaleCorrect":true,"scaleNote":"bricks correctly sized — ~27 courses visible matches expected 28 for 255cm wall","confidence":"high"}`;

interface Occluder {
  x: number; y: number; w: number; h: number;
  label: string; depth?: string;
}

interface WallBoundaries {
  ceilingLineY: number;
  floorLineY: number;
  leftEdgeX: number;
  rightEdgeX: number;
  selectionExceedsCeiling: boolean;
  selectionExceedsFloor: boolean;
  selectionExceedsLeftWall: boolean;
  selectionExceedsRightWall: boolean;
}

interface LightingInfo {
  primarySource: string;
  secondarySource?: string;
  temperature: string;
  temperatureKelvin: number;
  gradient: string;
  gradientIntensity: number;
  shadowType: string;
  shadowIntensity: number;
  ambientOcclusion: string;
  colorCast: string;
  exposure: string;
}

interface PerspectiveInfo {
  type: string;
  angleDeg: number;
  recedes: string;
  horizontalConvergence: string;
}

interface SceneAnalysis {
  wallHeightCm: number;
  wallWidthCm: number;
  ceilingHeightCm: number;
  pxPerCm: number;
  calibratedPxPerCm: number;
  referenceObjects: { name: string; realHeightCm: number; pxPerCm?: number; confidence: string }[];
  occluders: Occluder[];
  wallBoundaries: WallBoundaries;
  lighting: LightingInfo;
  perspective: PerspectiveInfo;
  textureScaleCorrect: boolean;
  scaleNote: string;
  confidence: string;
  // Legacy compat
  lightDirection: string;
  lightType: string;
  lightTemperature: string;
  lightIntensity: string;
  lightGradient: string;
  shadowIntensity: string;
  colorCast: string;
  perspectiveType: string;
  viewAngleDeg: number;
  horizontalConvergence: string;
}

const DEFAULT_ANALYSIS: SceneAnalysis = {
  wallHeightCm: 260,
  wallWidthCm: 350,
  ceilingHeightCm: 265,
  pxPerCm: 2.0,
  calibratedPxPerCm: 2.0,
  referenceObjects: [],
  occluders: [],
  wallBoundaries: {
    ceilingLineY: 0.0, floorLineY: 1.0, leftEdgeX: 0.0, rightEdgeX: 1.0,
    selectionExceedsCeiling: false, selectionExceedsFloor: false,
    selectionExceedsLeftWall: false, selectionExceedsRightWall: false,
  },
  lighting: {
    primarySource: "diffuse ambient", temperature: "neutral", temperatureKelvin: 5000,
    gradient: "even", gradientIntensity: 0.1, shadowType: "soft-diffuse",
    shadowIntensity: 0.3, ambientOcclusion: "subtle", colorCast: "none", exposure: "correct",
  },
  perspective: { type: "frontal", angleDeg: 0, recedes: "none", horizontalConvergence: "none" },
  textureScaleCorrect: true,
  scaleNote: "",
  confidence: "low",
  lightDirection: "diffuse",
  lightType: "ambient",
  lightTemperature: "neutral",
  lightIntensity: "moderate",
  lightGradient: "even",
  shadowIntensity: "medium",
  colorCast: "none",
  perspectiveType: "frontal",
  viewAngleDeg: 0,
  horizontalConvergence: "none",
};

async function analyzeScene(
  genAI: GoogleGenerativeAI,
  originalBuf: Buffer,
  compositeBuf: Buffer,
  maskOverlayBuf: Buffer,
  polyHeightFraction: number,
  polyWidthFraction: number,
  imgW: number,
  imgH: number,
): Promise<SceneAnalysis> {
  const textModelId = process.env.GEMINI_TEXT_MODEL || "gemini-2.5-pro";
  console.log(`[render-final] Step 1 model: ${textModelId}`);

  const heuristicH = Math.round(265 * polyHeightFraction * 1.05);
  const heuristicW = Math.round(265 * polyWidthFraction * (imgW / imgH) * 1.05);
  const fallback: SceneAnalysis = { ...DEFAULT_ANALYSIS, wallHeightCm: heuristicH, wallWidthCm: heuristicW };

  try {
    const model = genAI.getGenerativeModel({ model: textModelId });
    const result = await geminiRetry(() => model.generateContent([
      { text: SCENE_ANALYSIS_PROMPT },
      { inlineData: { mimeType: "image/jpeg", data: originalBuf.toString("base64") } },
      { inlineData: { mimeType: "image/jpeg", data: compositeBuf.toString("base64") } },
      { inlineData: { mimeType: "image/jpeg", data: maskOverlayBuf.toString("base64") } },
    ]));
    const text = result.response.text?.() ?? "";
    const clean = text.replace(/```(?:json)?\s*/g, "").replace(/```/g, "").trim();
    const m = clean.match(/\{[\s\S]*\}/);
    if (!m) return fallback;

    const parsed = JSON.parse(m[0]);
    const lt = parsed.lighting || {};
    const persp = parsed.perspective || {};
    const wb = parsed.wallBoundaries || {};

    const analysis: SceneAnalysis = {
      wallHeightCm: clamp(Number(parsed.wallHeightCm) || 260, 30, 1200),
      wallWidthCm: clamp(Number(parsed.wallWidthCm) || 350, 30, 2000),
      ceilingHeightCm: clamp(Number(parsed.ceilingHeightCm) || 265, 200, 500),
      pxPerCm: Number(parsed.pxPerCm || parsed.calibratedPxPerCm) || 2.0,
      calibratedPxPerCm: Number(parsed.calibratedPxPerCm || parsed.pxPerCm) || 2.0,
      referenceObjects: Array.isArray(parsed.referenceObjects) ? parsed.referenceObjects : [],
      occluders: Array.isArray(parsed.occluders) ? parsed.occluders : [],
      wallBoundaries: {
        ceilingLineY: Number(wb.ceilingLineY) || 0.0,
        floorLineY: Number(wb.floorLineY) || 1.0,
        leftEdgeX: Number(wb.leftEdgeX) || 0.0,
        rightEdgeX: Number(wb.rightEdgeX) || 1.0,
        selectionExceedsCeiling: !!wb.selectionExceedsCeiling,
        selectionExceedsFloor: !!wb.selectionExceedsFloor,
        selectionExceedsLeftWall: !!wb.selectionExceedsLeftWall,
        selectionExceedsRightWall: !!wb.selectionExceedsRightWall,
      },
      lighting: {
        primarySource: String(lt.primarySource || parsed.lightDirection || "diffuse"),
        secondarySource: lt.secondarySource,
        temperature: String(lt.temperature || parsed.lightTemperature || "neutral"),
        temperatureKelvin: Number(lt.temperatureKelvin) || 5000,
        gradient: String(lt.gradient || parsed.lightGradient || "even"),
        gradientIntensity: clamp(Number(lt.gradientIntensity) || 0.1, 0, 1),
        shadowType: String(lt.shadowType || "soft-diffuse"),
        shadowIntensity: clamp(Number(lt.shadowIntensity || parsed.shadowIntensity) || 0.3, 0, 1),
        ambientOcclusion: String(lt.ambientOcclusion || "subtle"),
        colorCast: String(lt.colorCast || parsed.colorCast || "none"),
        exposure: String(lt.exposure || "correct"),
      },
      perspective: {
        type: String(persp.type || parsed.perspectiveType || "frontal"),
        angleDeg: clamp(Number(persp.angleDeg || parsed.viewAngleDeg) || 0, 0, 80),
        recedes: String(persp.recedes || "none"),
        horizontalConvergence: String(persp.horizontalConvergence || parsed.horizontalConvergence || "none"),
      },
      textureScaleCorrect: parsed.textureScaleCorrect !== false,
      scaleNote: String(parsed.scaleNote || ""),
      confidence: String(parsed.confidence || "medium"),
      // Legacy compat
      lightDirection: String(lt.primarySource || parsed.lightDirection || "diffuse"),
      lightType: String(lt.primarySource || parsed.lightType || "ambient"),
      lightTemperature: String(lt.temperature || parsed.lightTemperature || "neutral"),
      lightIntensity: String(lt.exposure || parsed.lightIntensity || "moderate"),
      lightGradient: String(lt.gradient || parsed.lightGradient || "even"),
      shadowIntensity: String(lt.shadowType || parsed.shadowIntensity || "medium"),
      colorCast: String(lt.colorCast || parsed.colorCast || "none"),
      perspectiveType: String(persp.type || parsed.perspectiveType || "frontal"),
      viewAngleDeg: clamp(Number(persp.angleDeg || parsed.viewAngleDeg) || 0, 0, 80),
      horizontalConvergence: String(persp.horizontalConvergence || parsed.horizontalConvergence || "none"),
    };

    if (analysis.wallHeightCm > analysis.ceilingHeightCm + 10) {
      analysis.wallHeightCm = analysis.ceilingHeightCm;
    }

    console.log(`[render-final] Scene analysis: wall=${analysis.wallHeightCm}×${analysis.wallWidthCm}cm, ` +
      `light=${analysis.lighting.primarySource}/${analysis.lighting.temperature}, ` +
      `perspective=${analysis.perspective.type} ${analysis.perspective.angleDeg}°, ` +
      `occluders=${analysis.occluders.map(o => o.label).join(",")}, ` +
      `refs=${analysis.referenceObjects.map((r: { name: string }) => r.name).join(",")}, confidence=${analysis.confidence}`);

    return analysis;
  } catch (e) {
    console.warn("[render-final] Scene analysis failed:", e);
    return fallback;
  }
}

/* ══════════════════════════════════════════════════════════
   Texture scale computation
   Uses AI-measured wall dimensions + product physical specs
   ══════════════════════════════════════════════════════════ */

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
  dimensionSource: "ai" | "heuristic";
}

function computeTextureScale(
  polyHeightPx: number,
  polyWidthPx: number,
  textureHeightPx: number,
  textureWidthPx: number,
  meta: Record<string, unknown>,
  sceneAnalysis: SceneAnalysis,
): ScaleReport {
  const moduleH = Number(meta.moduleHeightMm || 65);
  const moduleW = Number(meta.moduleWidthMm || 245);
  const joint = Number(meta.jointMm || 10);
  const tileH = Number(meta.tileHeightMm) || 0;
  const tileW = Number(meta.tileWidthMm) || 0;
  const layout = String(meta.layoutType || "running-bond");
  let courses = Number(meta.albedoBrickCourses || meta.albedoCourses || 8);
  courses = clamp(courses, 1, 32);
  const isBrick = layout === "running-bond" || layout === "stretcher-bond";
  const isStone = layout === "random-stone";

  const courseH = moduleH + joint;
  const unitW = moduleW + joint;

  // Use tile dimensions from metadata if available, otherwise calculate
  let albedoHMm: number;
  let albedoWMm: number;
  if (tileH > 0 && tileW > 0) {
    albedoHMm = tileH;
    albedoWMm = tileW;
  } else if (isBrick || isStone) {
    albedoHMm = courseH * courses;
    const bricksPerRow = Math.max(1, Math.round(textureWidthPx / textureHeightPx * courses));
    albedoWMm = unitW * bricksPerRow;
  } else {
    const planks = clamp(Number(meta.albedoStackPlanks || meta.albedoPlankCount || 1), 1, 12);
    albedoHMm = moduleH * planks + joint * Math.max(planks - 1, 0);
    albedoWMm = moduleW;
  }

  // Wall dimensions from AI measurement (mm)
  const wallHMm = sceneAnalysis.wallHeightCm * 10;
  const wallWMm = sceneAnalysis.wallWidthCm * 10;
  const dimSource = sceneAnalysis.confidence !== "low" ? "ai" : "heuristic";

  // How many courses/units should fit in the wall
  const expectedCoursesH = wallHMm / courseH;
  const expectedUnitsW = wallWMm / unitW;

  // Compute scale: pixels per mm of real wall
  const pxPerMm = polyHeightPx / wallHMm;
  const targetAlbedoHPx = albedoHMm * pxPerMm;
  const scale = targetAlbedoHPx / textureHeightPx;

  // Verify: actual courses at this scale
  const tileHPx = textureHeightPx * scale;
  const coursesInPoly = (polyHeightPx / tileHPx) * courses;

  const tileWPx = textureWidthPx * scale;
  const albedoRepeatsW = polyWidthPx / tileWPx;
  const bricksPerAlbedo = (isBrick || isStone)
    ? Math.max(1, Math.round(albedoWMm / unitW))
    : 1;
  const unitsInPolyRow = albedoRepeatsW * bricksPerAlbedo;

  // If AI says scale looks wrong, try to correct
  let finalScale = scale;
  let verified = true;
  if (!sceneAnalysis.textureScaleCorrect && sceneAnalysis.confidence !== "low") {
    const hRatio = coursesInPoly / expectedCoursesH;
    if (hRatio < 0.7 || hRatio > 1.3) {
      const targetTileH = (polyHeightPx / expectedCoursesH) * courses;
      finalScale = targetTileH / textureHeightPx;
      verified = false;
    }
  }

  const mult = clamp(Number(meta.textureScaleMultiplier || 1), 0.5, 2.0);
  finalScale = Math.max(0.02, finalScale * mult);

  return {
    scale: finalScale,
    wallHeightMm: Math.round(wallHMm),
    wallWidthMm: Math.round(wallWMm),
    expectedCoursesH: Math.round(expectedCoursesH),
    expectedUnitsW: Math.round(expectedUnitsW),
    courseHeightMm: courseH,
    unitWidthMm: unitW,
    coursesInPoly: Math.round((polyHeightPx / (textureHeightPx * finalScale)) * courses),
    unitsInPolyRow: Math.round(unitsInPolyRow),
    verified,
    dimensionSource: dimSource,
  };
}

/* ══════════════════════════════════════════════════════════
   STEP 2: Gemini 3 Pro — Photorealistic Render
   Generates final image based on scene analysis + composite
   ══════════════════════════════════════════════════════════ */

function buildRenderPrompt(
  productName: string,
  meta: Record<string, unknown>,
  scene: SceneAnalysis,
  scaleReport: ScaleReport,
): string {
  const moduleH = Number(meta.moduleHeightMm || 80);
  const moduleW = Number(meta.moduleWidthMm || 245);
  const joint = Number(meta.jointMm || 10);
  const layout = String(meta.layoutType || "running-bond");
  const category = String(meta.category || "brick");
  const materialType = String(meta.materialType || "decorative cladding");
  const roughness = Number(meta.roughness || 0.7);
  const bumpDepth = Number(meta.bumpDepthMm || 5);
  const specular = Number(meta.specularIntensity || 0.05);
  const surfaceDesc = String(meta.surfaceDescription || "matte surface with subtle texture");

  const isBrick = layout === "running-bond" || layout === "stretcher-bond";
  const isStone = layout === "random-stone";
  const isWood = category === "wood";
  const unitLabel = isStone ? "stone" : isWood ? "panel" : "brick";
  const jointLabel = isStone ? "mortar" : isWood ? "gap" : "mortar joint";
  const courseLabel = isStone ? "stone rows" : isWood ? "panel rows" : "brick courses";
  const shadowEdge = isBrick || isStone ? "bottom and right edges" : "gap edges";

  const courseH = moduleH + joint;
  const unitW = moduleW + joint;
  const coursesInWall = Math.round((scene.wallHeightCm * 10) / courseH);
  const unitsInRow = Math.round((scene.wallWidthCm * 10) / unitW);

  const refsText = scene.referenceObjects.length > 0
    ? scene.referenceObjects.map((r: { name: string; realHeightCm?: number }) =>
      `${r.name} (~${r.realHeightCm || "?"}cm)`).join(", ")
    : "estimated from room proportions";

  const occludersList = scene.occluders.length > 0
    ? scene.occluders.map(o => `   - ${o.label} (${o.depth || "in front of wall"})`).join("\n")
    : "   (No major foreground objects detected — but verify against ORIGINAL)";

  const lt = scene.lighting;
  const persp = scene.perspective;

  const surfaceRendering = roughness > 0.7
    ? `ROUGH MATTE (${(roughness*100).toFixed(0)}%). ZERO specular highlights. Surface: ${surfaceDesc}`
    : roughness > 0.5
    ? `Near-matte (${(roughness*100).toFixed(0)}%). No specular spots. Surface: ${surfaceDesc}`
    : `Satin sheen (${(roughness*100).toFixed(0)}%, ${(specular*100).toFixed(0)}% specular). Surface: ${surfaceDesc}`;

  const tempEffect = lt.temperature.includes("warm")
    ? "Warm light tints the wall slightly orange/yellow — apply subtly"
    : lt.temperature.includes("cool")
    ? "Cool light tints the wall slightly blue — apply subtly"
    : "Neutral light — minimal color shift";

  const perspText = persp.type === "frontal" || persp.angleDeg < 10
    ? `Wall is nearly head-on (~${persp.angleDeg}°). Lines straight and parallel.`
    : `Wall at ~${persp.angleDeg}° angle, receding ${persp.recedes}. Lines converge ${persp.horizontalConvergence}.`;

  return `You are a photorealistic PHOTO EDITOR. Your job is to take an existing COMPOSITE image and make the textured wall area look like a real photograph. This is IMAGE EDITING — you are NOT generating a new image from scratch.

YOU RECEIVE 5 IMAGES (in order):
  1. ORIGINAL — unmodified room photograph (ground truth for everything outside the wall)
  2. COMPOSITE — room with "${productName}" texture already placed at CORRECT real-world scale
  3. MASK — ORIGINAL with ORANGE highlight = target wall area
  4. TEXTURE TILE — "${productName}" material swatch (color/detail reference only)
  5. ORIGINAL again (for direct comparison)

╔══════════════════════════════════════════════════════════════════╗
║  #1 CRITICAL RULE — PATTERN LOCK                                ║
║                                                                  ║
║  The COMPOSITE (image 2) has the SCIENTIFICALLY CORRECT texture ║
║  pattern. You MUST preserve it EXACTLY:                          ║
║                                                                  ║
║  → COUNT the ${unitLabel}s in the COMPOSITE: ~${scaleReport.unitsInPolyRow} across  ║
║  → COUNT the ${courseLabel}: ~${scaleReport.coursesInPoly} vertically              ║
║  → Each ${unitLabel} is ${moduleW}×${moduleH} mm (${(moduleW/10).toFixed(1)}×${(moduleH/10).toFixed(1)} cm)  ║
║  → ${jointLabel} gap: ${joint} mm                                ║
║                                                                  ║
║  Your output MUST have the IDENTICAL number of ${unitLabel}s     ║
║  at the IDENTICAL size. If the COMPOSITE shows ${scaleReport.unitsInPolyRow} ${unitLabel}s ║
║  across, your output shows ${scaleReport.unitsInPolyRow}. NOT fewer. NOT wider.    ║
║                                                                  ║
║  DO NOT re-draw, re-tile, or re-imagine the texture.            ║
║  DO NOT make ${unitLabel}s wider, narrower, taller, or shorter. ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  #2 WALL ONLY — never ceiling, floor, adjacent walls,      ║
║  furniture, or objects in front of the wall.                ║
╚══════════════════════════════════════════════════════════════╝

═══ YOUR TASK (photo editing, NOT generation) ═══

Start with the COMPOSITE image. Apply ONLY these photorealistic edits:
  1. Match the ORIGINAL's lighting/shadows onto the textured wall
  2. Add surface relief (micro-shadows on ${shadowEdge}, recessed ${jointLabel}s)
  3. Blend wall edges naturally (thin AO shadow lines at junctions)
  4. Restore any foreground objects that overlap the wall from ORIGINAL
  5. Keep everything outside the wall pixel-identical to ORIGINAL

You are NOT allowed to:
  ✗ Change the number of ${unitLabel}s (COMPOSITE has ~${scaleReport.unitsInPolyRow} across — KEEP IT)
  ✗ Change the width or height of any ${unitLabel}
  ✗ Re-draw or re-generate the texture pattern
  ✗ Change the ${jointLabel} spacing or width
  ✗ Zoom, crop, pan, reframe, or change dimensions

═══ FOREGROUND OBJECTS (restore from ORIGINAL) ═══

${occludersList}
These appear IN FRONT of the texture — copy them exactly from ORIGINAL on top of the textured wall.

═══ PRODUCT DIMENSIONS ═══

PRODUCT: ${materialType} — "${productName}"
  • Each ${unitLabel}: ${moduleW}×${moduleH} mm, ${jointLabel}: ${joint} mm
  • Course height: ${courseH} mm

WALL (measured using: ${refsText}):
  • ${scene.wallHeightCm} cm × ${scene.wallWidthCm} cm
  • COMPOSITE shows: ~${scaleReport.coursesInPoly} ${courseLabel} × ~${scaleReport.unitsInPolyRow} ${unitLabel}s/row
  • A door (200cm) = ~${Math.round(2000/courseH)} courses. Switch (115cm) = ~${Math.round(1150/courseH)} courses from floor.

═══ SURFACE PHYSICS ═══

${surfaceRendering}
Relief: ${bumpDepth} mm → micro-shadows on ${shadowEdge}
${isBrick || isStone ? `${jointLabel}s recessed ${Math.min(joint, 8)}mm — in shadow` : `Gaps are ${joint}mm deep dark channels`}

═══ LIGHTING (copy from ORIGINAL) ═══

  • Primary: ${lt.primarySource} (~${lt.temperatureKelvin}K ${lt.temperature})
  • Gradient: ${lt.gradient} (${(lt.gradientIntensity*100).toFixed(0)}%)
  • Shadows: ${lt.shadowType}, ${(lt.shadowIntensity*100).toFixed(0)}%
  • AO: ${lt.ambientOcclusion}
  • Color cast: ${lt.colorCast}
  • ${tempEffect}

═══ PERSPECTIVE ═══

${perspText}

═══ FINAL VERIFICATION (check before output) ═══

  ☐ COUNT ${unitLabel}s in your output — matches COMPOSITE (~${scaleReport.unitsInPolyRow} across)?
  ☐ Each ${unitLabel} same width as in COMPOSITE?
  ☐ Texture on wall ONLY — not ceiling/floor/other walls?
  ☐ ALL foreground objects from ORIGINAL preserved?
  ☐ Brightness gradient matches ORIGINAL?
  ☐ Same resolution/aspect — NO crop/zoom/pan?

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

/* ── Gemini retry helper ── */

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
   POST handler — 2-step pipeline
   Step 1: Gemini 2.5 Pro → scene analysis (cm-precise)
   Step 2: Gemini 3 Pro   → photorealistic render
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

    /* ── 5. Compute polygon bounds ── */
    const polyXs = scaledPoly.map(p => p.x);
    const polyYs = scaledPoly.map(p => p.y);
    let polyWidth = Math.max(...polyXs) - Math.min(...polyXs);
    let polyHeight = Math.max(...polyYs) - Math.min(...polyYs);
    const polyHeightFraction = polyHeight / H;
    const polyWidthFraction = polyWidth / W;
    let polyYMax = Math.round(Math.max(...polyYs));
    let polyXMin = Math.round(Math.min(...polyXs));

    /* ── 6. Initial texture tiling (pre-analysis, heuristic scale) ── */
    const t1 = Date.now();
    const texMeta = await sharp(textureFileBuffer).metadata();

    // Heuristic first pass for the composite that Gemini will analyze
    const heuristicAnalysis: SceneAnalysis = {
      ...DEFAULT_ANALYSIS,
      wallHeightCm: Math.round(265 * polyHeightFraction * 1.05),
      wallWidthCm: Math.round(265 * polyWidthFraction * (W / H) * 1.05),
    };

    let scaleReport = computeTextureScale(
      polyHeight, polyWidth, texMeta.height!, texMeta.width!, meta, heuristicAnalysis,
    );

    function buildTiledComposite(report: ScaleReport) {
      let tileW = Math.max(Math.round(texMeta.width! * report.scale), 1);
      let tileH = Math.max(Math.round(texMeta.height! * report.scale), 1);
      let tilesX = Math.ceil((W + tileW) / tileW);
      let tilesY = Math.ceil((H + tileH) / tileH);
      if (tilesX * tilesY > MAX_TILES) {
        const factor = Math.sqrt((tilesX * tilesY) / MAX_TILES);
        tileW = Math.round(tileW * factor);
        tileH = Math.round(tileH * factor);
      }
      return { tileW, tileH };
    }

    let { tileW, tileH } = buildTiledComposite(scaleReport);
    let scaledTexBuf = await sharp(textureFileBuffer).resize(tileW, tileH).toBuffer();

    async function buildBottomAnchoredTile(tw: number, th: number, texBuf: Buffer, anchorY: number, anchorX: number): Promise<Buffer> {
      const yPhase = th > 1 ? (th - (anchorY % th)) % th : 0;
      const xPhase = tw > 1 ? (tw - (anchorX % tw)) % tw : 0;
      const canvasW = W + tw;
      const canvasH = H + th;
      const ops: sharp.OverlayOptions[] = [];
      for (let y = 0; y < canvasH; y += th) {
        for (let x = 0; x < canvasW; x += tw) {
          ops.push({ input: texBuf, left: x, top: y });
        }
      }
      return sharp({
        create: { width: canvasW, height: canvasH, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 255 } },
      }).composite(ops)
        .extract({ left: xPhase, top: yPhase, width: W, height: H })
        .png().toBuffer();
    }

    let tiledBuffer = await buildBottomAnchoredTile(tileW, tileH, scaledTexBuf, polyYMax, polyXMin);

    /* ── 7. Polygon mask ── */
    const points = scaledPoly.map(p => `${Math.round(p.x)},${Math.round(p.y)}`).join(" ");
    const maskSvg = Buffer.from(
      `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">` +
      `<polygon points="${points}" fill="white"/>` +
      `</svg>`
    );
    let maskPng = await sharp(maskSvg).ensureAlpha().png().toBuffer();
    let grayMaskPng = await sharp(maskSvg).grayscale().png().toBuffer();

    let maskedTexture = await sharp(tiledBuffer)
      .composite([{ input: maskPng, blend: "dest-in" }])
      .png()
      .toBuffer();

    let compositeBuffer = await sharp(resizedImgBuffer)
      .composite([{ input: maskedTexture }])
      .jpeg({ quality: 92 })
      .toBuffer();

    /* ── 8. Mask overlay (orange highlight) ── */
    const overlaySvg = Buffer.from(
      `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">` +
      `<polygon points="${points}" fill="rgba(255,140,0,0.55)"/>` +
      `</svg>`
    );
    const overlayPng = await sharp(overlaySvg).png().toBuffer();
    let maskOverlayBuffer = await sharp(resizedImgBuffer)
      .composite([{ input: overlayPng }])
      .jpeg({ quality: 88 })
      .toBuffer();

    timings.initial_composite = Math.round((Date.now() - t1) / 100) / 10;

    /* ── 9. STEP 1: Gemini 2.5 Pro — Scene Analysis ── */
    const t2 = Date.now();
    const apiKey = process.env.GEMINI_API_KEY;
    let refinedB64: string;
    let geminiModel = "not-configured";

    if (apiKey) {
      const genAI = new GoogleGenerativeAI(apiKey);

      const scene = await analyzeScene(
        genAI, resizedImgBuffer, compositeBuffer, maskOverlayBuffer,
        polyHeightFraction, polyWidthFraction, W, H,
      );
      timings.analysis = Math.round((Date.now() - t2) / 100) / 10;

      /* ── 10. Refine mask with wall boundaries from analysis ── */
      const t3 = Date.now();
      const wb = scene.wallBoundaries;
      if (wb) {
        const origPolyTop = Math.min(...polyYs);
        const origPolyBottom = Math.max(...polyYs);
        const origPolyLeft = Math.min(...polyXs);
        const origPolyRight = Math.max(...polyXs);

        const wallTop = wb.selectionExceedsCeiling && wb.ceilingLineY > 0.01
          ? Math.max(0, Math.round(wb.ceilingLineY * H)) : 0;
        const wallBottom = wb.selectionExceedsFloor && wb.floorLineY < 0.99
          ? Math.min(H, Math.round(wb.floorLineY * H)) : H;
        const wallLeft = wb.selectionExceedsLeftWall && wb.leftEdgeX > 0.01
          ? Math.max(0, Math.round(wb.leftEdgeX * W)) : 0;
        const wallRight = wb.selectionExceedsRightWall && wb.rightEdgeX < 0.99
          ? Math.min(W, Math.round(wb.rightEdgeX * W)) : W;

        const refTop = Math.max(origPolyTop, wallTop);
        const refBottom = Math.min(origPolyBottom, wallBottom);
        const refLeft = Math.max(origPolyLeft, wallLeft);
        const refRight = Math.min(origPolyRight, wallRight);

        if (refBottom > refTop + 10 && refRight > refLeft + 10
            && (wallTop > 0 || wallBottom < H || wallLeft > 0 || wallRight < W)) {
          console.log(`[render-final] Refining mask with wall boundaries: ` +
            `top ${Math.round(origPolyTop)}→${Math.round(refTop)}, ` +
            `bottom ${Math.round(origPolyBottom)}→${Math.round(refBottom)}, ` +
            `left ${Math.round(origPolyLeft)}→${Math.round(refLeft)}, ` +
            `right ${Math.round(origPolyRight)}→${Math.round(refRight)}`);

          const refPoints = `${Math.round(refLeft)},${Math.round(refTop)} ` +
            `${Math.round(refRight)},${Math.round(refTop)} ` +
            `${Math.round(refRight)},${Math.round(refBottom)} ` +
            `${Math.round(refLeft)},${Math.round(refBottom)}`;
          const refMaskSvg = Buffer.from(
            `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">` +
            `<polygon points="${refPoints}" fill="white"/>` +
            `</svg>`
          );
          maskPng = await sharp(refMaskSvg).ensureAlpha().png().toBuffer();
          grayMaskPng = await sharp(refMaskSvg).grayscale().png().toBuffer();
          polyYMax = Math.round(refBottom);
          polyXMin = Math.round(refLeft);
          polyHeight = refBottom - refTop;
          polyWidth = refRight - refLeft;

          const refOverlaySvg = Buffer.from(
            `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">` +
            `<polygon points="${refPoints}" fill="rgba(255,140,0,0.55)"/>` +
            `</svg>`
          );
          const refOverlayPng = await sharp(refOverlaySvg).png().toBuffer();
          maskOverlayBuffer = await sharp(resizedImgBuffer)
            .composite([{ input: refOverlayPng }])
            .jpeg({ quality: 88 })
            .toBuffer();
        }
      }

      /* ── 11. Re-tile with AI-measured dimensions ── */
      const aiScaleReport = computeTextureScale(
        polyHeight, polyWidth, texMeta.height!, texMeta.width!, meta, scene,
      );

      const scaleDiff = Math.abs(aiScaleReport.scale - scaleReport.scale) / scaleReport.scale;
      const maskWasRefined = wb && (
        wb.selectionExceedsCeiling || wb.selectionExceedsFloor ||
        wb.selectionExceedsLeftWall || wb.selectionExceedsRightWall
      );
      if ((scaleDiff > 0.1 && scene.confidence !== "low") || maskWasRefined) {
        if (scaleDiff > 0.1) {
          console.log(`[render-final] Re-tiling: scale changed ${scaleReport.scale.toFixed(4)} → ${aiScaleReport.scale.toFixed(4)} (${(scaleDiff * 100).toFixed(1)}% diff)`);
        }
        scaleReport = aiScaleReport;

        const retiled = buildTiledComposite(scaleReport);
        tileW = retiled.tileW;
        tileH = retiled.tileH;
        scaledTexBuf = await sharp(textureFileBuffer).resize(tileW, tileH).toBuffer();

        tiledBuffer = await buildBottomAnchoredTile(tileW, tileH, scaledTexBuf, polyYMax, polyXMin);
        maskedTexture = await sharp(tiledBuffer)
          .composite([{ input: maskPng, blend: "dest-in" }])
          .png()
          .toBuffer();
        compositeBuffer = await sharp(resizedImgBuffer)
          .composite([{ input: maskedTexture }])
          .jpeg({ quality: 92 })
          .toBuffer();
      }

      console.log("[render-final] Final scale report:", JSON.stringify(scaleReport));
      timings.retile = Math.round((Date.now() - t3) / 100) / 10;

      const compositeB64 = "data:image/jpeg;base64," + compositeBuffer.toString("base64");

      /* ── 11. STEP 2: Gemini — Photorealistic Render ── */
      const t4 = Date.now();
      const imageModelId = process.env.GEMINI_IMAGE_MODEL || "gemini-3-pro-image-preview";
      geminiModel = imageModelId;
      console.log(`[render-final] Step 2 model: ${imageModelId}`);

      refinedB64 = compositeB64;
      let stage2Buf: Buffer | null = null;
      try {
        const imageModel = genAI.getGenerativeModel({
          model: imageModelId,
          generationConfig: { temperature: 0.1 } as Record<string, unknown>,
        });
        const prompt = buildRenderPrompt(meta.name || product_id, meta, scene, scaleReport);
        const result = await geminiRetry(() => imageModel.generateContent([
          { text: prompt },
          { inlineData: { mimeType: "image/jpeg", data: resizedImgBuffer.toString("base64") } },
          { inlineData: { mimeType: "image/jpeg", data: compositeBuffer.toString("base64") } },
          { inlineData: { mimeType: "image/jpeg", data: maskOverlayBuffer.toString("base64") } },
          { inlineData: { mimeType: "image/jpeg", data: textureFileBuffer.toString("base64") } },
          { inlineData: { mimeType: "image/jpeg", data: resizedImgBuffer.toString("base64") } },
        ]));

        const parts = result.response.candidates?.[0]?.content?.parts;
        const imagePart = parts?.find((p: { inlineData?: { mimeType: string; data: string } }) => p.inlineData);

        if (imagePart?.inlineData) {
          const renderedBuf = Buffer.from(imagePart.inlineData.data, "base64");
          const safeOutput = await maskedComposite(resizedImgBuffer, renderedBuf, grayMaskPng, W, H);
          stage2Buf = safeOutput;
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

      /* ── 12. STEP 3: Verification & Correction ── */
      const skipVerify = process.env.SKIP_VERIFICATION === "1" || process.env.SKIP_VERIFICATION === "true";
      if (stage2Buf && !skipVerify) {
        const t5 = Date.now();
        try {
          const verifyModel = genAI.getGenerativeModel({
            model: imageModelId,
            generationConfig: { temperature: 0.1 } as Record<string, unknown>,
          });

          const VERIFY_PROMPT = `You are a photorealistic rendering QA engine. Compare a rendered image with the original photograph and fix ANY issues.

YOU RECEIVE 4 IMAGES:
  1. ORIGINAL — unmodified room photograph (ground truth)
  2. RENDERED — AI-generated visualization (to be corrected)
  3. MASK — ORANGE highlight showing the target wall area
  4. TEXTURE TILE — the material being applied

VERIFICATION CHECKLIST — FIX EVERY ISSUE:

1. FOREGROUND OCCLUSION: Compare ORIGINAL with RENDERED object by object. Is ANY furniture/frame/switch/lamp now covered by texture? → Restore it EXACTLY as in ORIGINAL. Texture goes BEHIND objects.

2. BOUNDARY CHECK: Has texture leaked onto CEILING, FLOOR, or ADJACENT WALL? → Remove it, show original surface.

3. LIGHTING: Does brightness gradient match ORIGINAL? Are contact shadows present? Is ambient occlusion natural at junctions? → Fix mismatches.

4. TEXTURE: Correct physical size? Consistent pattern? Colors match TILE? → Preserve pattern, only fix issues.

5. EDGES: Smooth transitions? Thin shadow lines at junctions? → Fix harsh cuts.

RULES: Same resolution/aspect. Everything outside wall = pixel-identical to ORIGINAL. If no issues, return RENDERED unchanged.

Output ONLY the corrected image. No text.`;

          const verifyResult = await geminiRetry(() => verifyModel.generateContent([
            { text: VERIFY_PROMPT },
            { inlineData: { mimeType: "image/jpeg", data: resizedImgBuffer.toString("base64") } },
            { inlineData: { mimeType: "image/jpeg", data: stage2Buf!.toString("base64") } },
            { inlineData: { mimeType: "image/jpeg", data: maskOverlayBuffer.toString("base64") } },
            { inlineData: { mimeType: "image/jpeg", data: textureFileBuffer.toString("base64") } },
          ]));

          const vParts = verifyResult.response.candidates?.[0]?.content?.parts;
          const vImage = vParts?.find((p: { inlineData?: { mimeType: string; data: string } }) => p.inlineData);
          if (vImage?.inlineData) {
            const verifiedBuf = Buffer.from(vImage.inlineData.data, "base64");
            const safeFinal = await maskedComposite(resizedImgBuffer, verifiedBuf, grayMaskPng, W, H);
            refinedB64 = `data:image/jpeg;base64,${safeFinal.toString("base64")}`;
            console.log("[render-final] Verification pass completed successfully");
          } else {
            console.log("[render-final] Verification returned no image — keeping stage 2 result");
          }
        } catch (e) {
          console.warn("[render-final] Verification pass failed (non-fatal):", e);
        }
        timings.verification = Math.round((Date.now() - t5) / 100) / 10;
      }

      /* ── 13. Watermark ── */
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
          wallHeightCm: scene.wallHeightCm,
          wallWidthCm: scene.wallWidthCm,
          courses: scaleReport.coursesInPoly,
          unitsPerRow: scaleReport.unitsInPolyRow,
          dimensionSource: scaleReport.dimensionSource,
          confidence: scene.confidence,
          referenceObjects: scene.referenceObjects.map((r: { name: string }) => r.name),
        },
        analysis: {
          occluders: scene.occluders.map(o => o.label),
          lighting: scene.lighting,
          perspective: scene.perspective,
          wallBoundaries: scene.wallBoundaries,
        },
      });
    }

    // No API key — composite only
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
