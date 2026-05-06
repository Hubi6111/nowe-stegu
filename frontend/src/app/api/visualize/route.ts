import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";

const GEMINI_MODEL = "gemini-2.5-flash-image";

function stripDataUrl(base64: string): { data: string; mimeType: string } {
  const match = base64.match(/^data:([^;]+);base64,(.+)$/);
  if (match) return { mimeType: match[1], data: match[2] };
  return { mimeType: "image/jpeg", data: base64 };
}

export async function POST(req: NextRequest) {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "GEMINI_API_KEY is not configured." }, { status: 500 });
    }

    const body = await req.json();
    const {
      roomImageBase64,       // original room photo
      compositeImageBase64,  // room + texture already composited client-side
      maskImageBase64,       // white = wall, black = keep
      textureName,
      textureDescription,
      materialType,
      wallWidthPx,
      wallHeightPx,
      imageWidthPx,
      imageHeightPx,
    } = body;

    if (!roomImageBase64 || !compositeImageBase64 || !maskImageBase64) {
      return NextResponse.json({ error: "Missing required image data" }, { status: 400 });
    }

    const roomImg = stripDataUrl(roomImageBase64);
    const compositeImg = stripDataUrl(compositeImageBase64);
    const maskImg = stripDataUrl(maskImageBase64);

    const wallAreaPct = Math.round((wallWidthPx * wallHeightPx) / (imageWidthPx * imageHeightPx) * 100);

    const prompt = `You are a professional interior design photo editor. I need you to refine a wall texture composite.

I am providing 3 images in this exact order:

1. **ORIGINAL ROOM** — The unmodified room photograph for lighting/shadow reference.
2. **COMPOSITE** — The same room photo but with the wall texture "${textureName}"${materialType ? ` (${materialType})` : ""} already pasted onto the wall area. The texture pattern, scale, and tiling are ALREADY CORRECT and pixel-perfect. DO NOT change the texture pattern, scale, brick layout, or colors.
3. **MASK** — White = wall area with texture. Black = unchanged areas.

YOUR TASK (refinement only):
- Start from the COMPOSITE image (image 2) as your base
- KEEP the texture EXACTLY as it appears in the composite — same pattern, same scale, same brick/stone layout, same colors. Do NOT re-generate, re-tile, or modify the texture in any way
- The texture MUST stay STRICTLY within the white mask boundary — do NOT let it bleed outside
- Adjust the texture's brightness, contrast, and color temperature to match the room's existing lighting
- Add subtle shadows and light effects matching the room's ambient light direction
- Any objects in FRONT of the wall (furniture, plants, lamps, shelves, decorations) that are visible in the ORIGINAL image (image 1) must be RESTORED on top of the textured wall — they should appear naturally in front
- Blend the texture edges smoothly with adjacent surfaces (floor, ceiling, other walls)
- The wall covers ~${wallAreaPct}% of the image
${textureDescription ? `- Surface material: ${textureDescription}` : ""}

CRITICAL: Do NOT modify the texture pattern or scale. The brick/stone/wood layout is already perfect. Only adjust lighting and restore foreground objects.

OUTPUT: Return ONLY the refined photograph. Same dimensions as input. No text.`;

    const ai = new GoogleGenAI({ apiKey });

    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: [{
        role: "user",
        parts: [
          { text: prompt },
          { inlineData: { mimeType: roomImg.mimeType, data: roomImg.data } },
          { inlineData: { mimeType: compositeImg.mimeType, data: compositeImg.data } },
          { inlineData: { mimeType: maskImg.mimeType, data: maskImg.data } },
        ],
      }],
      config: {
        responseModalities: ["TEXT", "IMAGE"],
        temperature: 0.2,
      },
    });

    const candidates = response.candidates;
    if (!candidates || candidates.length === 0) {
      return NextResponse.json({ error: "No response from Gemini model" }, { status: 500 });
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
      return NextResponse.json({
        error: "Gemini nie zwrócił obrazu. " + (textResponse || "Spróbuj ponownie."),
      }, { status: 500 });
    }

    return NextResponse.json({
      image: `data:${resultMimeType};base64,${resultImageBase64}`,
      text: textResponse,
    });
  } catch (err: unknown) {
    console.error("Visualize API error:", err);
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
