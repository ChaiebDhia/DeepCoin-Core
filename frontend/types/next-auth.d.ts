/**
 * types/next-auth.d.ts
 * =====================
 * TypeScript module augmentation for next-auth v5.
 *
 * WHY augment the Session and JWT interfaces:
 *   Next-auth's default Session.user only has { name?, email?, image? }.
 *   We store additional fields (id, role, access_token, display_name).
 *   Without this augmentation, TypeScript would report errors when accessing
 *   session.user.role or session.user.access_token.
 *
 * HOW module augmentation works:
 *   TypeScript's "declaration merging" lets you add properties to an existing
 *   interface by re-declaring it inside a `declare module` block.
 *   This is the official next-auth v5 pattern (not a hack).
 *
 * WHERE these fields come from:
 *   auth.config.ts jwt callback copies them from the authorize() return value.
 *   auth.config.ts session callback copies them into session.user.
 */

import type { DefaultSession, DefaultJWT } from "next-auth";

declare module "next-auth" {
  /**
   * Returned by useSession(), auth(), and passed to the session callback.
   * We extend it with our custom fields.
   */
  interface Session {
    user: {
      /** PostgreSQL UUID from the users table */
      id: string;
      /** RBAC role: "admin" | "curator" | "analyst" */
      role: string;
      /** Optional display name (Dr. Ahmed Chaieb, etc.) */
      display_name: string | null;
      /** FastAPI JWT access token — used in Authorization: Bearer header */
      access_token: string;
      /**
       * Unix timestamp (ms) when the access_token expires.
       * Set on first login by auth.config.ts jwt callback.
       * Used by the Axios refresh interceptor to schedule proactive refresh.
       */
      access_expires_at?: number;
    } & DefaultSession["user"];
  }

  /** Returned by the User object from Credentials.authorize() */
  interface User {
    role?:             string;
    display_name?:     string | null;
    access_token?:     string;
    /** FastAPI access token TTL in seconds — used to compute access_expires_at */
    expires_in?:       number;
  }
}

declare module "next-auth/jwt" {
  /** Stored in the encrypted next-auth cookie. */
  interface JWT extends DefaultJWT {
    id?:                string;
    role?:              string;
    display_name?:      string | null;
    access_token?:      string;
    /** Unix timestamp (ms) — when the current access_token expires (-60s buffer applied) */
    access_expires_at?: number;
  }
}
