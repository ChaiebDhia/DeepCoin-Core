/**
 * app/api/auth/refresh-access-token/route.ts
 * ============================================
 * Next.js Route Handler — proxies POST /auth/refresh to FastAPI and relays
 * the rotated httpOnly refresh-token cookie back to the browser.
 *
 * ── Why a Next.js proxy instead of the browser calling FastAPI directly? ──
 *
 * FastAPI's refresh token lives in an httpOnly cookie with:
 *   - path="/auth"   → only sent to /auth/* endpoints
 *   - samesite="lax" → cross-origin sub-resource requests are blocked
 *
 * In development, FastAPI runs on port 8000 and Next.js on port 3000.
 * They are different origins. The browser CANNOT attach the httpOnly
 * cookie to a direct POST http://127.0.0.1:8000/auth/refresh cross-origin
 * call because SameSite=Lax blocks it for programmatic fetches.
 *
 * In production (Docker Compose + Nginx), everything is on the same
 * origin (example.com). A direct browser→FastAPI call would work there.
 * But Dev ≠ Prod here, which means bugs that only appear in prod — the
 * worst kind.
 *
 * The proxy solves both environments:
 *   Browser → POST /api/auth/refresh-access-token  (same origin, Next.js)
 *             └→ cookies forwarded automatically by the browser
 *   Next.js server → POST http://api:8000/auth/refresh  (Docker internal)
 *             └→ Cookie header forwarded manually from the incoming request
 *   FastAPI → issues new access token + rotated refresh cookie (set-cookie)
 *   Next.js → relays the new set-cookie header back to the browser
 *
 * ── Refresh token rotation (reuse detection) ──
 *
 * FastAPI REVOKES the incoming refresh token on every call and issues a NEW
 * one. If an attacker steals and uses the refresh token, the legitimate user's
 * next refresh will find their token already revoked → 401 → forced re-login.
 * The Next.js proxy MUST forward the new set-cookie so the browser's cookie
 * jar is updated with the rotated token. Failing to forward it would log out
 * the user on every silent refresh.
 *
 * ── Error responses ──
 *
 * 401 → refresh token expired / revoked → client redirects to /login
 * 503 → FastAPI is unreachable            → client retries or shows error
 */

import { NextRequest, NextResponse } from "next/server";

const FASTAPI_URL = process.env.AUTH_FASTAPI_URL ?? "http://127.0.0.1:8000";

export async function POST(req: NextRequest): Promise<NextResponse> {
  // ── Forward the browser's cookies to FastAPI ─────────────────────────────
  //
  // The browser sends ALL cookies for this origin with the request. We
  // extract the full Cookie header and pass it to FastAPI so the httpOnly
  // deepcoin_refresh_token cookie reaches the /auth/refresh endpoint.
  const cookieHeader = req.headers.get("cookie") ?? "";

  // ── Call FastAPI ──────────────────────────────────────────────────────────
  let fastapiRes: Response;
  try {
    fastapiRes = await fetch(`${FASTAPI_URL}/auth/refresh`, {
      method:  "POST",
      headers: {
        "Cookie":       cookieHeader,
        "Content-Type": "application/json",
      },
      // No body — FastAPI reads the refresh token from the cookie
    });
  } catch {
    // FastAPI is unreachable (container down, network issue)
    return NextResponse.json(
      { detail: "Cannot reach the authentication server. Please try again." },
      { status: 503 },
    );
  }

  // ── Non-200 from FastAPI ──────────────────────────────────────────────────
  if (!fastapiRes.ok) {
    const body = await fastapiRes.json().catch(() => ({
      detail: "Token refresh failed. Please log in again.",
    }));
    return NextResponse.json(body, { status: fastapiRes.status });
  }

  // ── Success: relay new access token + rotated refresh cookie ─────────────
  const data = await fastapiRes.json() as {
    access_token: string;
    expires_in:   number;
    token_type:   string;
  };

  const nextRes = NextResponse.json({
    access_token: data.access_token,
    expires_in:   data.expires_in,
  });

  // CRITICAL: relay the Set-Cookie header so the browser's cookie jar is
  // updated with the rotated refresh token. Without this, the old (revoked)
  // refresh token stays in the browser and the very next refresh call fails.
  const setCookie = fastapiRes.headers.get("set-cookie");
  if (setCookie) {
    nextRes.headers.set("set-cookie", setCookie);
  }

  return nextRes;
}
