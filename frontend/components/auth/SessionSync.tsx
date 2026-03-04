"use client";

/**
 * components/auth/SessionSync.tsx
 * =================================
 * Zero-render bridge: watches the NextAuth session and writes the JWT into
 * the module-level cache inside lib/api.ts.
 *
 * WHAT: Invisible component (renders null). Calls setAuthToken() whenever
 *       the NextAuth session changes — on login, logout, or token refresh.
 *
 * WHY this solves ClientFetchError:
 *   The previous approach called getSession() inside every Axios interceptor.
 *   getSession() from next-auth/react fires fetch(/api/auth/session) on each
 *   call. When that endpoint is slow or unavailable, NextAuth fires
 *   console.error("ClientFetchError") INTERNALLY before throwing — so our
 *   try/catch in the interceptor could not suppress that log.
 *
 *   This component uses useSession() (a React hook) which reads from the
 *   already-hydrated SessionProvider context — ZERO network calls per request.
 *   The token is written once to the module-level cache when the session loads,
 *   and updated synchronously on sign-in / sign-out.
 *
 * HOW: Import this inside <Providers> (providers.tsx), after <SessionProvider>.
 *   It must be a Client Component because useSession() is a hook.
 *   It renders null — completely invisible to the UI.
 */

import { useEffect }    from "react";
import { useSession }   from "next-auth/react";
import { setAuthToken, setSessionUpdateFn } from "@/lib/api";

export function SessionSync() {
  const { data: session, update } = useSession();

  // Sync the access_token into the module-level Axios cache.
  // Runs whenever the session object changes (login, logout, token update).
  useEffect(() => {
    const token =
      (session?.user as { access_token?: string } | undefined)?.access_token ??
      null;
    setAuthToken(token);
  }, [session]);

  // Register the NextAuth update() function so the Axios response interceptor
  // can push a new access_token into the session cookie after a silent refresh.
  // WHY cleanup on unmount: avoids dangling reference if the component
  // (inside <SessionProvider>) ever unmounts before the app does.
  useEffect(() => {
    setSessionUpdateFn(update as (data: Record<string, unknown>) => void);
    return () => setSessionUpdateFn(null);
  }, [update]);

  return null;
}
