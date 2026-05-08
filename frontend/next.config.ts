import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals = [...(config.externals || []), { canvas: "canvas" }];
    }
    return config;
  },
  // Allow large request bodies for image uploads (4 images in two-stage pipeline)
  serverExternalPackages: [],
  // Increase body size limit for API routes (default 1MB is not enough for multiple images)
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
