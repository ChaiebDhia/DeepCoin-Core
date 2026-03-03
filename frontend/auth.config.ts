/**
 * auth.config.ts
 * ==============
 * Next-Auth v5 base configuration — Edge-compatible (no Node.js-only APIs).
 *
 * WHY two files (auth.config.ts + auth.ts):
 *   Next.js middleware runs in the Edge runtime (V8 isolate, not Node.js).
 *   Some next-auth providers and adapters use Node.js APIs (crypto, fs, etc.)
 *   that are unavailable at the Edge. The pattern is:
 *     - auth.config.ts: Edge-safe config (providers + callbacks + pages)
 *     - auth.ts:        Full NextAuth() init — can use Node.js APIs; exports
 *                       { auth, handlers, signIn, signOut }
 *     - middleware.ts:  Imports auth from auth.ts for route protection
 *
 * WHY credentials provider + FastAPI (not database adapter):
 *   Auth is owned by FastAPI (bcrypt, JWT, refresh-token rotation, RBAC).
 *   Next-Auth is used ONLY as a session layer: it stores { access_token, user }
 *   in a signed httpOnly cookie so server components can read auth state.
 *   Duplicating password hashing in Next.js would split auth logic across two
 *   layers — an anti-pattern. Credentials provider delegates to FastAPI.
 *
 * WHY session strategy "jwt" (not "database"):
 *   We don't want a second session table (FastAPI already maintains
 *   refresh_tokens). JWT sessions are stateless; the session data is stored
 *   encrypted in the browser cookie. ACCESS_TOKEN is short-lived (15 min);
 *   if the user's FastAPI token expires, the next classify call returns 401
 *   and the frontend calls /auth/refresh transparently.
 */

import type { NextAuthConfig } from "next-auth";
import Credentials from "next-auth/providers/credentials";

// ── FastAPI base URL for server-side auth calls ───────────────────────────────
//
// In development:  AUTH_FASTAPI_URL=http://127.0.0.1:8000  (set in .env.local)
// In Docker:       AUTH_FASTAPI_URL=http://api:8000         (set in docker-compose.yml)
// Fallback:        http://127.0.0.1:8000                    (dev default)
//
// WHY NOT use NEXT_PUBLIC_CLASSIFY_URL:
//   NEXT_PUBLIC_* variables are inlined at BUILD time and sent to the browser.
//   This URL is server-side only (credentials authorize() runs on the server).
//   A server-only env var is never exposed to the client bundle.
const FASTAPI_URL = process.env.AUTH_FASTAPI_URL ?? "http://127.0.0.1:8000";

export const authConfig: NextAuthConfig = {
  /**
   * Session strategy: jwt
   *
   * The session is stored in a signed & encrypted httpOnly cookie.
   * No session table needed in Next.js's DB — FastAPI owns persistence.
   * maxAge: 3600 s (1 hour) matches FastAPI's access token expiry.
   */
  session: {
    strategy: "jwt",
    maxAge:   3600,     // 1 hour — matches FastAPI access token TTL
  },

  /**
   * Custom pages — avoid next-auth's built-in /api/auth/signin page.
   * Our /login page has the DeepCoin design system.
   */
  pages: {
    signIn:  "/login",
    signOut: "/",
    error:   "/login",  // next-auth appends ?error=... to this URL
  },

  providers: [
    /**
     * Credentials provider — delegates authentication to FastAPI.
     *
     * HOW the flow works:
     *   1. User submits email + password in LoginForm
     *   2. LoginForm calls signIn("credentials", { email, password })
     *   3. Next-Auth calls authorize() (server-side, never sent to browser)
     *   4. authorize() POSTs to FastAPI /auth/login
     *   5. If FastAPI returns 200 + access_token → return user object
     *   6. Next-Auth creates a signed cookie containing the JWT + user fields
     *   7. Browser stores the cookie; useSession() can read it anywhere
     *
     * WHY not forward the password to the cookie:
     *   We only store what we need: id, email, role, display_name, access_token.
     *   The raw password is never persisted beyond the authorize() call.
     */
    Credentials({
      name: "DeepCoin",
      credentials: {
        email:    { label: "Email",    type: "email" },
        password: { label: "Password", type: "password" },
      },

      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) return null;

        try {
          const res = await fetch(`${FASTAPI_URL}/auth/login`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
              email:    credentials.email,
              password: credentials.password,
            }),
            // next-auth authorize runs in Node.js runtime — fetch is available
            // We set a generous 10 s timeout; FastAPI login is fast (<100 ms)
            signal: AbortSignal.timeout(10_000),
          });

          if (!res.ok) {
            if (res.status === 403) {
              // Account exists but is not yet verified / suspended.
              // throw causes result.error = "CallbackRouteError" in the browser,
              // which LoginForm distinguishes from a plain bad-credentials 401.
              const body: { detail?: string } = await res.json().catch(() => ({}));
              throw new Error(body.detail ?? "Please verify your email before signing in.");
            }
            // 401 — wrong email or password → return null → CredentialsSignin
            return null;
          }

          const data = await res.json() as {
            access_token:  string;
            token_type:    string;
            user: {
              id:           string;
              email:        string;
              role:         string;
              display_name: string | null;
              status:       string;
            };
          };

          // next-auth stores this object in the JWT cookie
          return {
            id:           data.user.id,
            email:        data.user.email,
            name:         data.user.display_name ?? data.user.email.split("@")[0],
            role:         data.user.role,
            display_name: data.user.display_name,
            access_token: data.access_token,
          };
        } catch {
          // Network error (FastAPI down, timeout, etc.) → treat as auth failure
          return null;
        }
      },
    }),
  ],

  callbacks: {
    /**
     * jwt callback — called when the JWT is created or updated.
     *
     * WHY copy user fields into the token here:
     *   On first sign-in, `user` is populated by the authorize() return value.
     *   On subsequent requests, `user` is undefined — we read from the existing
     *   token. This pattern persists role + access_token across page navigations.
     */
    async jwt({ token, user }) {
      if (user) {
        // First sign-in: persist custom fields from authorize() return value
        token.id           = user.id as string;
        token.role         = (user as { role?: string }).role ?? "analyst";
        token.display_name = (user as { display_name?: string | null }).display_name ?? null;
        token.access_token = (user as { access_token?: string }).access_token ?? "";
      }
      return token;
    },

    /**
     * session callback — shapes what useSession() returns in the browser.
     *
     * WHY expose access_token in the session:
     *   The browser-side Axios interceptor in lib/api.ts reads
     *   session.user.access_token and adds it as Authorization: Bearer to
     *   every API call. This is the bridge between next-auth sessions and
     *   FastAPI's JWT validation.
     *
     * WHY not store the token in localStorage:
     *   httpOnly cookie (managed by next-auth) is immune to XSS.
     *   Exposing access_token in the session object is safe because:
     *     - The client already holds the cookie that contains it
     *     - The session endpoint is protected by CSRF token
     */
    async session({ session, token }) {
      if (token && session.user) {
        session.user.id           = token.id as string;
        session.user.role         = token.role as string;
        session.user.display_name = token.display_name as string | null;
        session.user.access_token = token.access_token as string;
      }
      return session;
    },
  },
};
