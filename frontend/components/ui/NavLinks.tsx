"use client";

/**
 * components/ui/NavLinks.tsx
 * ==========================
 * Client-side navigation links that are aware of auth status.
 *
 * WHAT: Renders the main nav links in the header. The "Analyse" link
 *       is auth-gated: unauthenticated users are sent to the login page
 *       with a callbackUrl so they land on /analyse after signing in.
 *
 * WHY a separate client component (not inline in header.tsx):
 *   header.tsx is a Server Component — it contains no React hooks.
 *   useSession() is a React hook that requires "use client" context.
 *   Extracting just the nav into a client component keeps the rest of
 *   the header server-rendered and avoids shipping unnecessary JS.
 *
 * WHY redirect to login instead of /analyse directly:
 *   /analyse runs the full classification pipeline and hits the FastAPI
 *   backend. Authentication ensures requests are associated with a user,
 *   results are saved to history, and the classify endpoint can apply
 *   per-user rate limiting. The login redirect is UX, not a security
 *   gate — but it sets the right expectation for new visitors.
 *
 * BEHAVIOUR during loading:
 *   While useSession() hydrates, "Analyse" defaults to the login redirect.
 *   This is the safest fallback — an unauthenticated click goes to login,
 *   never to /analyse unexpectedly.
 */

import { useSession } from "next-auth/react";
import Link           from "next/link";

/* ── Shared link style ────────────────────────────────────────────────── */

const linkCls =
  "px-3 py-1.5 rounded-md text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--surface-2)] transition-colors";

/* ── Component ────────────────────────────────────────────────────────── */

export function NavLinks() {
  const { data: session } = useSession();

  /**
   * If the user is authenticated, "Analyse" goes straight to /analyse.
   * If not (or while loading), it goes to the login page with a callbackUrl
   * so NextAuth redirects them back to /analyse after a successful sign-in.
   */
  const analyseHref = session ? "/analyse" : "/login?callbackUrl=/analyse";

  return (
    <nav className="flex items-center gap-1">
      <Link href="/#features"  className={linkCls}>Features</Link>
      <Link href={analyseHref} className={linkCls}>Analyse</Link>
      <Link href="/history"    className={linkCls}>History</Link>
    </nav>
  );
}
