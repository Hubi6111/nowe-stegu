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
      roomImageBase64,
      compositeImageBase64,
      textureName,
    } = body;

    if (!roomImageBase64 || !compositeImageBase64) {
      return NextResponse.json({ error: "Missing required image data" }, { status: 400 });
    }

    const roomImg = stripDataUrl(roomImageBase64);
    const compositeImg = stripDataUrl(compositeImageBase64);

    const prompt = `Return the COMPOSITE image (image 2) with ONLY these minimal changes:

I provide 2 images:
1. ORIGINAL ROOM — unmodified photo, use ONLY as lighting reference.
2. COMPOSITE — the room with texture "${textureName}" already applied on the wall. This is your output base.

RULES — follow ALL of them strictly:
- Output the COMPOSITE image as-is with at most a subtle brightness/warmth shift so the texture matches the room's ambient light.
- DO NOT add, remove, invent, or hallucinate ANY object, shadow, reflection, furniture, plant, lamp, person, or detail that is not already present in the COMPOSITE.
- DO NOT repaint, re-tile, distort, blur, sharpen, or re-generate the texture. Every pixel of the texture must stay exactly as it appears in the COMPOSITE.
- DO NOT change anything outside the textured wall area. Every non-wall pixel must be identical to the COMPOSITE.
- DO NOT add artistic effects, vignettes, grain, noise, lens flare, or color grading.
- If in doubt, change NOTHING. Less editing is always better.

OUTPUT: Return ONLY the image. Same dimensions as COMPOSITE. No text, no explanation.`;

    const ai = new GoogleGenAI({ apiKey });

    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: [{
        role: "user",
        parts: [
          { text: prompt },
          { inlineData: { mimeType: roomImg.mimeType, data: roomImg.data } },
          { inlineData: { mimeType: compositeImg.mimeType, data: compositeImg.data } },
        ],
      }],
      config: {
        responseModalities: ["TEXT", "IMAGE"],
        temperature: 0.0,
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
