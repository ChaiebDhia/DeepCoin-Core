/**
 * components/ui/NavLinks.tsx
 * ==========================
 * Public navigation links rendered in the site header.
 *
 * WHAT: Static links to public-facing pages visible to ALL visitors —
 *       logged-in or anonymous, no auth check required.
 *
 * WHY no auth logic here:
 *   Auth-gated actions (Analyse, History) belong in the UserMenu dropdown
 *   so they are only visible to authenticated users. Keeping this component
 *   pure-public makes the nav straightforward for new visitors and removes
 *   the need for "use client" + session hydration overhead in the header.
 *
 * WHY Server Component (no "use client"):
 *   All links are static. There are no React hooks or browser APIs.
 *   Shipping this as a Server Component means zero JS for the nav links —
 *   they render on the server and arrive as plain HTML.
 */

import Link from "next/link";

/* ── Shared link style ────────────────────────────────────────────────── */

const linkCls =
  "px-3 py-1.5 rounded-md text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--surface-2)] transition-colors";

/* ── Component ────────────────────────────────────────────────────────── */

export function NavLinks() {
  return (
    <nav className="flex items-center gap-1">
      <Link href="/#features" className={linkCls}>Features</Link>
      <Link href="/explore"   className={linkCls}>Explore</Link>
      <Link href="/chat"      className={linkCls}>AI Chat</Link>
      <Link href="/about"     className={linkCls}>About</Link>
      <Link href="/docs"      className={linkCls}>Docs</Link>
    </nav>
  );
}
