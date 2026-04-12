import type { NextConfig } from "next";

const traceAssets = [
  "../assets/textures/stegu/**/*",
  "../../assets/textures/stegu/**/*",
];
const traceWatermark = ["./public/watermark.png"];

/** Prefer IPv4 loopback — fewer connection issues than localhost → ::1 on some hosts. */
function normalizeApiProxyUrl(url: string): string {
  const u = url.replace(/\/$/, "");
  return u.replace(/^http:\/\/localhost\b/, "http://127.0.0.1");
}

const nextConfig: NextConfig = {
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals = [...(config.externals || []), { canvas: "canvas" }];
    }
    return config;
  },
  outputFileTracingIncludes: {
    "/api/products": traceAssets,
    "/api/textures/[productId]": traceAssets,
    "/api/render-final": [...traceAssets, ...traceWatermark],
  },
  async rewrites() {
    const apiUrl = process.env.API_PROXY_URL;
    if (!apiUrl) return { beforeFiles: [], afterFiles: [], fallback: [] };
    const base = normalizeApiProxyUrl(apiUrl);
    // fallback: Next.js checks ALL its own routes first (including dynamic [productId]).
    // Only unmatched /api/* requests fall through to FastAPI proxy.
    return {
      beforeFiles: [],
      afterFiles: [],
      fallback: [
        { source: "/api/:path*", destination: `${base}/api/:path*` },
      ],
    };
  },
};

export default nextConfig;
