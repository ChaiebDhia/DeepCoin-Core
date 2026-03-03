/**
 * middleware.ts
 * =============
 * Next.js middleware — runs on every matched request BEFORE the page renders.
 *
 * PURPOSE: Protect authenticated routes. If the user has no valid next-auth
 * session cookie, redirect them to /login.
 *
 * HOW it works with next-auth v5:
 *   next-auth exports `auth` as a middleware-compatible function.
 *   We re-export it directly for simple protection, or wrap it for custom logic.
 *   The `config.matcher` array controls which paths trigger the middleware.
 *
 * PROTECTED ROUTES:
 *   /history         — classification history (personal data, auth required)
 *   /history/*       — individual history detail pages
 *   /admin           — admin dashboard (extra role check inside the page)
 *   /admin/*         — all admin sub-pages
 *   /collections     — future: saved collections (auth required)
 *   /collections/*   — future: collection detail
 *
 * PUBLIC ROUTES (explicitly excluded from matcher):
 *   /                — classify page (guests can use it)
 *   /login           — login form
 *   /register        — registration form
 *   /api/*           — FastAPI proxy + next-auth API routes
 *   /_next/*         — Next.js static assets
 *
 * WHY exclude /api/* from the matcher:
 *   The /api/auth/[...nextauth] route is next-auth's own handler.
 *   Protecting it with auth middleware would create a redirect loop.
 *   FastAPI routes have their own authentication (Bearer token / X-API-Key).
 */

import { auth } from "./auth";
import { NextResponse } from "next/server";

export default auth((req) => {
  const { nextUrl, auth: session } = req;

  // If already authenticated, let the request through
  if (session) return NextResponse.next();

  // Not authenticated → redirect to /login with a `callbackUrl` param
  // so LoginForm can redirect back to the original page after login
  const loginUrl = new URL("/login", nextUrl.origin);
  loginUrl.searchParams.set("callbackUrl", nextUrl.pathname + nextUrl.search);
  return NextResponse.redirect(loginUrl);
});

export const config = {
  /**
   * Matcher — which routes trigger this middleware.
   *
   * WHY negative lookahead for _next and api:
   *   Middleware runs on every request that matches. We must exclude static
   *   assets (_next/static, _next/image) and API routes to avoid intercepting
   *   them. The negative lookahead syntax is the recommended Next.js pattern.
   */
  matcher: [
    "/history",
    "/history/:path*",
    "/analyse",               // Analyse page requires an account (tracks usage + stores history)
    "/admin",
    "/admin/:path*",
    "/collections",
    "/collections/:path*",
  ],
};
