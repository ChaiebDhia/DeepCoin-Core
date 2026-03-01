import type { NextConfig } from "next";

/**
 * next.config.ts
 * ==============
 * WHY rewrites instead of direct axios calls to localhost:8000:
 *   1. Avoids CORS preflight — browser sees same origin (/api/*)
 *   2. The API URL lives in one place (DEEPCOIN_API_URL env var)
 *   3. Works for both dev (localhost:8000) and prod (container DNS)
 *
 * WHY no bodyParser size config here:
 *   Rewrites pass the request through unchanged — Next.js never parses
 *   the multipart body, so the 10 MB coin upload reaches FastAPI intact.
 */
const nextConfig: NextConfig = {
  async rewrites() {
    const apiBase = process.env.DEEPCOIN_API_URL ?? "http://localhost:8000";
    return [
      {
        source:      "/api/:path*",
        destination: `${apiBase}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
