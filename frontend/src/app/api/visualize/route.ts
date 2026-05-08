import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { checkRateLimit, recordUsage, getClientIp } from "../rate-limit";

// Vercel serverless config — two-stage pipeline needs more time
export const maxDuration = 60; // seconds (max for Hobby plan)

// Allow larger request bodies for multiple images
export const config = {
  api: {
    bodyParser: {
      sizeLimit: "10mb",
    },
  },
};

/* ══════════════════════════════════════════════════════════════════════════════
   Two-stage pipeline:
     Stage 1 — Scene Analysis   (text-only, pro model)
     Stage 2 — Texture Render   (image output, flash-image model)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Stage 1: Analysis models (text-only, high reasoning).
 * Primary: gemini-3.1-pro-preview, fallback: gemini-2.5-pro
 */
const ANALYSIS_MODEL_CHAIN = [
  "gemini-3.1-pro-preview",
  "gemini-2.5-pro",
];

/**
 * Stage 2: Render models (image generation).
 * Primary: gemini-3.1-flash-image-preview, fallback: gemini-2.5-flash-image
 */
const RENDER_MODEL_CHAIN = [
  "gemini-3.1-flash-image-preview",
  "gemini-2.5-flash-image",
];

/* ── Helpers ── */

function stripDataUrl(base64: string): { data: string; mimeType: string } {
  const match = base64.match(/^data:([^;]+);base64,(.+)$/);
  if (match) return { mimeType: match[1], data: match[2] };
  return { mimeType: "image/jpeg", data: base64 };
}

function isRetryableError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const msg = err.message || "";
  return (
    msg.includes("503") ||
    msg.includes("UNAVAILABLE") ||
    msg.includes("429") ||
    msg.includes("RESOURCE_EXHAUSTED") ||
    msg.includes("overloaded")
  );
}

/* ══════════════════════════════════════════════════════════════════════════════
   STAGE 1 — Scene Analysis
   ══════════════════════════════════════════════════════════════════════════════ */

function buildAnalysisPrompt(
  textureName: string,
  textureDescription: string,
  materialType: string,
  wallWidthPx: number,
  wallHeightPx: number,
  imageWidthPx: number,
  imageHeightPx: number,
): string {
  return `You are a high-precision scene analysis engine for an interior wall visualizer.

INPUTS PROVIDED:
1. Original room photograph
2. Black-and-white mask image — white area = the selected wall surface
3. Texture info: name="${textureName}", type="${materialType}", description="${textureDescription}"
4. Wall region: ${wallWidthPx}×${wallHeightPx} px within a ${imageWidthPx}×${imageHeightPx} px image

YOUR TASK:
Analyze the scene and return a JSON object with precise data that a rendering engine will use to apply the texture realistically.

Analyze carefully:
- The real-world dimensions of the masked wall based on visual cues (door frames ~2.1m, windows, furniture, ceiling height ~2.5m standard)
- The perspective geometry — is the wall frontal, angled left, angled right?
- Vanishing points and how the texture grid should converge
- Lighting direction, intensity, and color temperature
- How many texture modules (tiles/bricks/panels) fit on this wall at correct real-world scale
- The wall surface orientation and normal direction

RESPOND WITH ONLY THIS JSON (no markdown, no code fences, no explanation):
{
  "estimatedWallWidthM": <number, estimated real width in meters>,
  "estimatedWallHeightM": <number, estimated real height in meters>,
  "moduleCountX": <number, how many texture modules fit horizontally>,
  "moduleCountY": <number, how many texture modules fit vertically>,
  "moduleScaleX": <number, horizontal scale factor for perspective, 1.0 = frontal>,
  "moduleScaleY": <number, vertical scale factor for perspective, 1.0 = frontal>,
  "perspectiveType": "<frontal | left-angled | right-angled | overhead>",
  "perspectiveDescription": "<brief description of wall angle and vanishing points>",
  "lightDirection": "<e.g. top-left, right, ambient>",
  "lightIntensity": "<low | medium | high>",
  "lightColorTemp": "<warm | neutral | cool>",
  "surfaceOrientation": "<vertical | horizontal | angled>",
  "wallNormalDirection": "<description of wall normal vector direction>",
  "shadowsPresent": <boolean, are there visible shadows on/near the wall>,
  "occludingObjects": "<description of any furniture/objects in front of the wall>",
  "recommendations": "<2-3 sentences with specific tips for realistic texture application>"
}`;
}

interface AnalysisResult {
  estimatedWallWidthM?: number;
  estimatedWallHeightM?: number;
  moduleCountX?: number;
  moduleCountY?: number;
  moduleScaleX?: number;
  moduleScaleY?: number;
  perspectiveType?: string;
  perspectiveDescription?: string;
  lightDirection?: string;
  lightIntensity?: string;
  lightColorTemp?: string;
  surfaceOrientation?: string;
  wallNormalDirection?: string;
  shadowsPresent?: boolean;
  occludingObjects?: string;
  recommendations?: string;
}

async function runAnalysisStage(
  ai: GoogleGenAI,
  roomImg: { data: string; mimeType: string },
  maskImg: { data: string; mimeType: string },
  textureName: string,
  textureDescription: string,
  materialType: string,
  wallWidthPx: number,
  wallHeightPx: number,
  imageWidthPx: number,
  imageHeightPx: number,
): Promise<{ analysis: AnalysisResult; model: string }> {
  const prompt = buildAnalysisPrompt(
    textureName,
    textureDescription,
    materialType,
    wallWidthPx,
    wallHeightPx,
    imageWidthPx,
    imageHeightPx,
  );

  const contentRequest = {
    contents: [
      {
        role: "user" as const,
        parts: [
          { text: prompt },
          { inlineData: { mimeType: roomImg.mimeType, data: roomImg.data } },
          { inlineData: { mimeType: maskImg.mimeType, data: maskImg.data } },
        ],
      },
    ],
    config: {
      responseModalities: ["TEXT" as const],
      temperature: 0.1,
    },
  };

  let lastError: unknown = null;

  for (const model of ANALYSIS_MODEL_CHAIN) {
    try {
      console.log(`[Stage 1 — Analysis] Trying model: ${model}`);
      const response = await ai.models.generateContent({
        model,
        ...contentRequest,
      });

      const candidates = response.candidates;
      if (!candidates || candidates.length === 0) {
        console.warn(`[Stage 1] ${model}: No candidates returned, trying next…`);
        continue;
      }

      const resParts = candidates[0].content?.parts || [];
      let textResponse = "";
      for (const part of resParts) {
        if (part.text) textResponse += part.text;
      }

      if (!textResponse.trim()) {
        console.warn(`[Stage 1] ${model}: Empty text response, trying next…`);
        continue;
      }

      // Parse JSON from response — strip code fences if present
      let jsonStr = textResponse.trim();
      const fenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (fenceMatch) jsonStr = fenceMatch[1].trim();

      let analysis: AnalysisResult;
      try {
        analysis = JSON.parse(jsonStr);
      } catch {
        console.warn(`[Stage 1] ${model}: Failed to parse JSON: ${jsonStr.substring(0, 200)}`);
        // Try to extract JSON object from the response
        const objMatch = jsonStr.match(/\{[\s\S]*\}/);
        if (objMatch) {
          try {
            analysis = JSON.parse(objMatch[0]);
          } catch {
            console.warn(`[Stage 1] ${model}: Second parse attempt failed, trying next model…`);
            continue;
          }
        } else {
          continue;
        }
      }

      console.log(`[Stage 1 — Analysis] ✅ Success with model: ${model}`);
      console.log(`[Stage 1 — Analysis] Result:`, JSON.stringify(analysis, null, 2));
      return { analysis, model };
    } catch (err: unknown) {
      lastError = err;
      const errMsg = err instanceof Error ? err.message : String(err);
      console.warn(`[Stage 1] ${model} failed: ${errMsg.substring(0, 200)}`);

      if (isRetryableError(err)) {
        console.log(`[Stage 1] Retryable error, trying next model…`);
        continue;
      }
      throw err;
    }
  }

  // All analysis models failed — throw
  const fallbackMsg = lastError instanceof Error ? lastError.message : "All analysis models unavailable";
  throw new Error(`Stage 1 Analysis failed: ${fallbackMsg}`);
}

/* ══════════════════════════════════════════════════════════════════════════════
   STAGE 2 — Texture Render
   ══════════════════════════════════════════════════════════════════════════════ */

function buildRenderPrompt(
  textureName: string,
  analysisJson: AnalysisResult,
): string {
  const analysisStr = JSON.stringify(analysisJson, null, 2);

  return `You are a high-precision image editing engine for an interior wall visualizer.

INPUTS:
1. Original room image
2. User-selected mask image (white = wall area, black = keep unchanged)
3. Texture image (the material to apply)
4. Pre-composited image (texture already roughly placed on wall — use as starting reference)
5. Analysis JSON from the scene-analysis stage:
${analysisStr}

TASK:
Apply the texture "${textureName}" ONLY inside the masked wall area and return a realistic final render.

STRICT INSTRUCTIONS:
1. Use the analysis JSON as authoritative guidance for:
   - estimated real-world scale
   - target wall dimensions
   - module count
   - module scale
   - perspective
   - lighting
   - orientation
2. Apply the texture only within the masked area.
3. Do not modify anything outside the mask.
4. Preserve:
   - furniture
   - floor
   - ceiling
   - windows
   - doors
   - decorations
   - shadows
   - reflections
   - camera framing
5. The texture must be scaled according to its real physical dimensions, not stretched arbitrarily to fill the area.
6. Repeat the texture naturally according to the calculated module count and orientation.
7. Follow the wall perspective exactly so the texture looks installed on the wall surface, not pasted on top.
8. Blend the texture with the original scene lighting:
   - preserve shadows
   - preserve contact shading
   - preserve depth
   - preserve room illumination
9. Keep the result photorealistic.
10. Return only the final edited image with no text explanation.`;
}

async function runRenderStage(
  ai: GoogleGenAI,
  roomImg: { data: string; mimeType: string },
  maskImg: { data: string; mimeType: string },
  textureImg: { data: string; mimeType: string },
  compositeImg: { data: string; mimeType: string },
  textureName: string,
  analysis: AnalysisResult,
): Promise<{ imageBase64: string; mimeType: string; model: string }> {
  const prompt = buildRenderPrompt(textureName, analysis);

  const contentRequest = {
    contents: [
      {
        role: "user" as const,
        parts: [
          { text: prompt },
          { inlineData: { mimeType: roomImg.mimeType, data: roomImg.data } },
          { inlineData: { mimeType: maskImg.mimeType, data: maskImg.data } },
          { inlineData: { mimeType: textureImg.mimeType, data: textureImg.data } },
          { inlineData: { mimeType: compositeImg.mimeType, data: compositeImg.data } },
        ],
      },
    ],
    config: {
      responseModalities: ["TEXT" as const, "IMAGE" as const],
      temperature: 0.0,
    },
  };

  let lastError: unknown = null;

  for (const model of RENDER_MODEL_CHAIN) {
    try {
      console.log(`[Stage 2 — Render] Trying model: ${model}`);
      const response = await ai.models.generateContent({
        model,
        ...contentRequest,
      });

      const candidates = response.candidates;
      if (!candidates || candidates.length === 0) {
        console.warn(`[Stage 2] ${model}: No candidates returned, trying next…`);
        continue;
      }

      const resParts = candidates[0].content?.parts || [];
      let resultImageBase64: string | null = null;
      let resultMimeType = "image/png";
      let textResponse = "";

      for (const part of resParts) {
        if (part.text) textResponse += part.text;
        else if (part.inlineData) {
          resultImageBase64 = part.inlineData.data || null;
          resultMimeType = part.inlineData.mimeType || "image/png";
        }
      }

      if (!resultImageBase64) {
        console.warn(`[Stage 2] ${model}: No image in response, trying next…`);
        if (textResponse) {
          console.warn(`[Stage 2] ${model}: Text response was: ${textResponse.substring(0, 200)}`);
        }
        continue;
      }

      console.log(`[Stage 2 — Render] ✅ Success with model: ${model}`);
      return { imageBase64: resultImageBase64, mimeType: resultMimeType, model };
    } catch (err: unknown) {
      lastError = err;
      const errMsg = err instanceof Error ? err.message : String(err);
      console.warn(`[Stage 2] ${model} failed: ${errMsg.substring(0, 200)}`);

      if (isRetryableError(err)) {
        console.log(`[Stage 2] Retryable error, trying next model…`);
        continue;
      }
      throw err;
    }
  }

  // All render models failed
  const fallbackMsg = lastError instanceof Error ? lastError.message : "All render models unavailable";
  throw new Error(`Stage 2 Render failed: ${fallbackMsg}`);
}

/* ══════════════════════════════════════════════════════════════════════════════
   POST handler — orchestrates both stages
   ══════════════════════════════════════════════════════════════════════════════ */

export async function POST(req: NextRequest) {
  const timings: Record<string, number> = {};
  const t0 = Date.now();

  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "GEMINI_API_KEY is not configured." }, { status: 500 });
    }

    // Rate limiting
    const clientIp = getClientIp(req);
    const rateCheck = checkRateLimit(clientIp);
    if (!rateCheck.allowed) {
      return NextResponse.json({
        error: `Przekroczono dzienny limit generacji (${rateCheck.limit}). Spróbuj ponownie jutro.`,
        remaining: 0,
        limit: rateCheck.limit,
      }, { status: 429 });
    }

    // Parse request body with error handling
    let body: Record<string, unknown>;
    try {
      body = await req.json();
    } catch {
      return NextResponse.json({ error: "Nieprawidłowy format żądania (JSON parse error)" }, { status: 400 });
    }

    const {
      roomImageBase64,
      compositeImageBase64,
      maskImageBase64,
      textureImageBase64,
      textureName,
      textureDescription,
      materialType,
      wallWidthPx,
      wallHeightPx,
      imageWidthPx,
      imageHeightPx,
    } = body as {
      roomImageBase64?: string;
      compositeImageBase64?: string;
      maskImageBase64?: string;
      textureImageBase64?: string;
      textureName?: string;
      textureDescription?: string;
      materialType?: string;
      wallWidthPx?: number;
      wallHeightPx?: number;
      imageWidthPx?: number;
      imageHeightPx?: number;
    };

    if (!roomImageBase64 || !compositeImageBase64) {
      return NextResponse.json({ error: "Missing required image data" }, { status: 400 });
    }

    const roomImg = stripDataUrl(roomImageBase64);
    const compositeImg = stripDataUrl(compositeImageBase64);
    const maskImg = maskImageBase64 ? stripDataUrl(maskImageBase64) : null;
    const textureImg = textureImageBase64 ? stripDataUrl(textureImageBase64) : null;

    const ai = new GoogleGenAI({ apiKey });

    // ─── STAGE 1: Scene Analysis ───
    console.log("═══════════════════════════════════════════════");
    console.log("[Pipeline] Starting Stage 1 — Scene Analysis");
    console.log("═══════════════════════════════════════════════");

    let analysis: AnalysisResult = {};
    let analysisModel = "none";

    if (maskImg) {
      try {
        const t1 = Date.now();
        const result = await runAnalysisStage(
          ai,
          roomImg,
          maskImg,
          textureName || "unknown",
          textureDescription || "",
          materialType || "",
          (wallWidthPx as number) || 0,
          (wallHeightPx as number) || 0,
          (imageWidthPx as number) || 0,
          (imageHeightPx as number) || 0,
        );
        analysis = result.analysis;
        analysisModel = result.model;
        timings.analysis = (Date.now() - t1) / 1000;
      } catch (err) {
        // Analysis failure is non-fatal — we proceed with empty analysis
        console.warn("[Pipeline] Stage 1 failed, proceeding with empty analysis:", err instanceof Error ? err.message : err);
      }
    } else {
      console.warn("[Pipeline] No mask provided, skipping analysis stage");
    }

    // ─── STAGE 2: Texture Render ───
    console.log("═══════════════════════════════════════════════");
    console.log("[Pipeline] Starting Stage 2 — Texture Render");
    console.log("═══════════════════════════════════════════════");

    // Use mask or create a dummy white mask for render
    const renderMaskImg = maskImg || { data: "", mimeType: "image/png" };
    // Use texture image if provided, otherwise use composite as fallback
    const renderTextureImg = textureImg || compositeImg;

    const t2 = Date.now();
    const renderResult = await runRenderStage(
      ai,
      roomImg,
      renderMaskImg,
      renderTextureImg,
      compositeImg,
      textureName || "texture",
      analysis,
    );
    timings.render = (Date.now() - t2) / 1000;
    timings.total = (Date.now() - t0) / 1000;

    console.log("═══════════════════════════════════════════════");
    console.log(`[Pipeline] ✅ Complete! Analysis: ${analysisModel}, Render: ${renderResult.model} (${timings.total}s)`);
    console.log("═══════════════════════════════════════════════");

    // Record usage for rate limiting & admin stats
    recordUsage(clientIp, textureName || "unknown", textureName || "unknown", renderResult.model, timings);

    return NextResponse.json({
      image: `data:${renderResult.mimeType};base64,${renderResult.imageBase64}`,
      analysis,
      analysisModel,
      renderModel: renderResult.model,
      remaining: rateCheck.remaining - 1,
    });
  } catch (err: unknown) {
    console.error("Visualize API error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";

    // Check if it's a retryable error for user messaging
    if (isRetryableError(err)) {
      return NextResponse.json({
        error: `Wszystkie modele AI są przeciążone. Spróbuj ponownie za chwilę. (${message.substring(0, 100)})`,
      }, { status: 503 });
    }

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
