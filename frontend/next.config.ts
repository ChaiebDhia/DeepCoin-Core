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
 *
 * Security headers — applied to all routes:
 *   X-Frame-Options:        Prevents clickjacking (this app is never embedded)
 *   X-Content-Type-Options: Prevents MIME sniffing (serve JPEG as JPEG, always)
 *   Referrer-Policy:        Only send origin on cross-origin navigations
 *   Permissions-Policy:     Deny camera/mic/geo access (unused browser APIs)
 *   Content-Security-Policy:
 *     blob: in img-src is required — CoinUploader uses createObjectURL for preview
 *     unsafe-inline in style-src is required — Tailwind generates inline styles
 *     unsafe-eval in script-src is removed in production (Next.js only needs it in dev)
 */

// Security header definitions — applied to every response
const isDev = process.env.NODE_ENV !== "production";

const securityHeaders = [
  { key: "X-DNS-Prefetch-Control",   value: "on" },
  { key: "X-Frame-Options",          value: "DENY" },
  { key: "X-Content-Type-Options",   value: "nosniff" },
  { key: "Referrer-Policy",          value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy",       value: "camera=(), microphone=(), geolocation=(), payment=()" },
  // P10 — HSTS: tell browsers to ONLY connect over HTTPS for the next 2 years.
  // max-age=63072000 (2 years) is the preload requirement. Ignored on HTTP (dev).
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains; preload" },
  {
    key:   "Content-Security-Policy",
    value: [
      "default-src 'self'",
      // P13 — unsafe-eval is needed by Next.js HMR in dev only.
      // Production builds do NOT use eval — removing it closes a real XSS vector.
      `script-src 'self'${isDev ? " 'unsafe-eval'" : ""} 'unsafe-inline'`,
      "style-src 'self' 'unsafe-inline'",
      // blob: required for URL.createObjectURL() coin image preview
      // data: required for base64 inline images
      "img-src 'self' blob: data:",
      "font-src 'self'",
      // connect-src self: Next.js HMR (dev) + /api/* proxied calls
      "connect-src 'self' http://127.0.0.1:8000 http://localhost:8000",
      "frame-ancestors 'none'",
      "base-uri 'self'",
      "form-action 'self'",
    ].join("; "),
  },
];

const nextConfig: NextConfig = {
  // Hide the Next.js dev overlay icons (build spinner bottom-right,
  // framework logo bottom-left). They are purely cosmetic in dev mode
  // and add visual clutter to the UI during testing.
  devIndicators: false,

  async headers() {
    return [
      {
        // Apply to all routes
        source:  "/:path*",
        headers: securityHeaders,
      },
    ];
  },

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
