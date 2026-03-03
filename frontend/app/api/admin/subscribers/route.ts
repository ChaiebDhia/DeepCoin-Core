/**
 * app/api/admin/subscribers/route.ts
 * ====================================
 * Server-side proxy for the subscriber list.
 *
 * WHAT: Validates the caller's NextAuth session, enforces admin/curator role,
 *       then calls FastAPI GET /api/subscribers (with the server-side API key)
 *       and returns the response to the browser.
 *
 * WHY a proxy instead of calling FastAPI directly from the browser:
 *   The DEEPCOIN_API_KEY is a server-side secret ($env var, not exposed to
 *   the browser). A Next.js Route Handler can read it at runtime on the
 *   server and forward authenticated requests — the key never reaches the
 *   client JS bundle.
 *
 * Authorization:
 *   - No session → 401
 *   - Session but role !== admin / curator → 403
 *   - Authorized → proxy FastAPI response transparently
 */

import { NextResponse } from "next/server";
import { auth }         from "../../../../auth";

type SessionUser = {
  email?:        string;
  display_name?: string;
  role?:         string;
  access_token?: string;
};

export async function GET() {
  // 1. Validate NextAuth session
  const session = await auth();
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // 2. Enforce admin / curator role
  const role = (session.user as SessionUser).role ?? "analyst";
  if (role !== "admin" && role !== "curator") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  // 3. Forward request to FastAPI with the server-side API key
  const apiBase = process.env.DEEPCOIN_API_URL  ?? "http://127.0.0.1:8000";
  const apiKey  = process.env.DEEPCOIN_API_KEY  ?? "";

  try {
    const res = await fetch(`${apiBase}/api/subscribers`, {
      headers: {
        "X-API-Key": apiKey,
      },
      // Don't cache — always return the freshest subscriber list
      cache: "no-store",
    });

    if (!res.ok) {
      // FastAPI is unreachable or returned an error — surface an empty list
      // rather than a 500 so the admin page degrades gracefully.
      return NextResponse.json([], { status: 200 });
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    // Network error (FastAPI not running, etc.)
    return NextResponse.json([], { status: 200 });
  }
}
