import { NextRequest, NextResponse } from "next/server";
import { getClientIp, getRemainingForIp } from "../rate-limit";

export async function GET(req: NextRequest) {
  const ip = getClientIp(req);
  const info = getRemainingForIp(ip);
  return NextResponse.json(info);
}
