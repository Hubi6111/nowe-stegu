const DAILY_LIMIT = 20;

/* ── In-memory store ──
   Persists within a single Vercel serverless instance (warm lambda).
   Resets on cold start — acceptable for rate-limiting and live stats.
   For full persistence, connect Upstash Redis / Vercel KV in the future. */

interface DailyEntry { date: string; count: number }
interface GenRecord {
  id: string;
  timestamp: string;
  client_ip: string;
  product_id: string;
  product_name: string;
  gemini_model: string;
  timings: Record<string, number>;
}

const usage = new Map<string, DailyEntry>();
const generations: GenRecord[] = [];
const byProduct = new Map<string, number>();
const byDay = new Map<string, number>();
let totalGenerations = 0;
const allIps = new Set<string>();

function todayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

function cleanOldEntries() {
  const today = todayKey();
  for (const [key, val] of usage) {
    if (val.date !== today) usage.delete(key);
  }
}

export function getClientIp(req: Request): string {
  const xff = req.headers.get("x-forwarded-for");
  if (xff) return xff.split(",")[0].trim();
  const real = req.headers.get("x-real-ip");
  if (real) return real.trim();
  return "unknown";
}

export function checkRateLimit(ip: string): { allowed: boolean; remaining: number; limit: number; used: number } {
  cleanOldEntries();
  const today = todayKey();
  const entry = usage.get(ip);
  if (!entry || entry.date !== today) {
    return { allowed: true, remaining: DAILY_LIMIT, limit: DAILY_LIMIT, used: 0 };
  }
  const remaining = Math.max(0, DAILY_LIMIT - entry.count);
  return { allowed: remaining > 0, remaining, limit: DAILY_LIMIT, used: entry.count };
}

export function recordUsage(ip: string, productId: string, productName: string, geminiModel: string, timings: Record<string, number>): void {
  const today = todayKey();
  const entry = usage.get(ip);
  if (!entry || entry.date !== today) {
    usage.set(ip, { date: today, count: 1 });
  } else {
    entry.count++;
  }

  allIps.add(ip);
  totalGenerations++;

  byProduct.set(productName, (byProduct.get(productName) || 0) + 1);
  byDay.set(today, (byDay.get(today) || 0) + 1);

  const record: GenRecord = {
    id: `gen-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: new Date().toISOString(),
    client_ip: ip,
    product_id: productId,
    product_name: productName,
    gemini_model: geminiModel,
    timings,
  };
  generations.unshift(record);
  // Keep last 500 in memory
  if (generations.length > 500) generations.length = 500;
}

export function getRemainingForIp(ip: string): { remaining: number; limit: number; used: number; unlimited: boolean } {
  cleanOldEntries();
  const today = todayKey();
  const entry = usage.get(ip);
  const used = entry && entry.date === today ? entry.count : 0;
  return { remaining: DAILY_LIMIT - used, limit: DAILY_LIMIT, used, unlimited: false };
}

/* ── Admin data accessors ── */

export function getStats() {
  const today = todayKey();
  const todayUsers: Record<string, number> = {};
  let todayTotal = 0;
  for (const [ip, entry] of usage) {
    if (entry.date === today) {
      todayUsers[ip] = entry.count;
      todayTotal += entry.count;
    }
  }

  return {
    total: totalGenerations,
    today: todayTotal,
    unique_ips: allIps.size,
    by_product: Object.fromEntries(byProduct),
    by_day: Object.fromEntries(byDay),
    daily_limit: DAILY_LIMIT,
    usage_today: { date: today, users: todayUsers, total: todayTotal },
  };
}

export function getGenerations(offset: number, limit: number, filterIp?: string) {
  let items = generations;
  if (filterIp) {
    items = items.filter(g => g.client_ip === filterIp);
  }
  const total = items.length;
  const sliced = items.slice(offset, offset + limit);
  return { total, offset, limit, items: sliced };
}

export function getLimitsData() {
  const today = todayKey();
  const todayUsers: Record<string, number> = {};
  let todayTotal = 0;
  for (const [ip, entry] of usage) {
    if (entry.date === today) {
      todayUsers[ip] = entry.count;
      todayTotal += entry.count;
    }
  }
  return {
    daily_limit: DAILY_LIMIT,
    usage: { date: today, users: todayUsers, total: todayTotal },
  };
}
