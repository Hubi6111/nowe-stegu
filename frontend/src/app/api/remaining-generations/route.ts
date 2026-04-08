import { NextResponse } from "next/server";

/**
 * GET /api/remaining-generations
 *
 * On Vercel (no FastAPI backend): returns unlimited.
 * Locally: the FastAPI backend handles this via next.config rewrites,
 * so this route is only reached when FastAPI is unavailable.
 */
export async function GET() {
  return NextResponse.json({
    remaining: -1,
    limit: 0,
    used: 0,
    unlimited: true,
  });
}
