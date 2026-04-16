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
 *     http://127.0.0.1:8000 in img-src is required — Grad-CAM PNGs are served
 *       directly from FastAPI (/api/gradcam/{filename}) via gradcamDisplayUrl()
 *       which bypasses the Next.js proxy (same pattern as PDF download links)
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
      // http://127.0.0.1:8000 required for Grad-CAM PNGs served directly from
      // FastAPI (/api/gradcam/{filename}).  gradcamDisplayUrl() bypasses the
      // Next.js proxy (same pattern as PDF downloads) so the browser fetches
      // the PNG straight from FastAPI — which is a cross-origin URL that the
      // CSP img-src directive must explicitly allow.
      "img-src 'self' blob: data: http://127.0.0.1:8000 http://localhost:8000",
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
  // output: "standalone" — produces .next/standalone for Docker deployments.
  // WHY: The standalone build traces all imports and copies only the code +
  //   node_modules the app actually uses.  The result is a self-contained
  //   Node.js server (~40 MB) versus the full source + node_modules (~300 MB).
  //   Required by frontend/Dockerfile Stage 3 runner (CMD node server.js).
  // NOTE: In dev (`next dev`) standalone output is ignored — only active on
  //   `next build`.  Development hot-reload works identically with or without it.
  output: "standalone",

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
    return {
      /**
       * beforeFiles: empty — do not intercept anything before route handlers.
       */
      beforeFiles: [],

      /**
       * afterFiles: empty — do not intercept anything after route handlers either.
       * WHY: keeping this empty is belt-and-suspenders for the Turbopack bug
       * described below.
       */
      afterFiles: [],

      /**
       * fallback: proxy /api/* → FastAPI only when NO Next.js route matches.
       *
       * WHY fallback instead of the previous plain-array (afterFiles):
       *   Next.js 15 Turbopack has a known ordering bug where "afterFiles"
       *   rewrites can fire before App Router route handlers are checked.
       *   The symptom: GET /api/auth/session is proxied to FastAPI (which returns
       *   {"detail":"Not Found"}) instead of being handled by NextAuth's catch-all
       *   at app/api/auth/[...nextauth]/route.ts.
       *
       *   "fallback" rewrites are the LAST thing checked — after ALL static files,
       *   all App Router handlers, all Pages Router pages, and all afterFiles
       *   rewrites. So NextAuth ALWAYS wins for /api/auth/**, and FastAPI ALWAYS
       *   wins for every other /api/** route that has no Next.js handler.
       *
       *   Routing resolution order (Next.js 15):
       *     1. headers
       *     2. redirects
       *     3. beforeFiles rewrites        ← empty
       *     4. public/ static files
       *     5. App Router route handlers   ← NextAuth handles /api/auth/**
       *     6. afterFiles rewrites         ← empty
       *     7. Dynamic routes
       *     8. fallback rewrites           ← FastAPI handles /api/health, /api/classify …
       */
      fallback: [
        {
          source:      "/api/:path*",
          destination: `${apiBase}/api/:path*`,
        },
      ],
    };
  },
};

import createNextIntlPlugin from 'next-intl/plugin';
const withNextIntl = createNextIntlPlugin('./src/i18n/request.ts');

export default withNextIntl(nextConfig);
