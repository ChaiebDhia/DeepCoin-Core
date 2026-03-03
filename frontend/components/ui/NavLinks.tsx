"use client";

/**
 * components/ui/NavLinks.tsx
 * ==========================
 * Desktop-only horizontal navigation pill strip (md and wider).
 * On mobile this renders nothing — the hamburger lives in MobileNav.tsx,
 * which is placed inside the right cluster so it sits flush with UserMenu.
 *
 * WHY separated: having the hamburger in the middle slot of the header
 * (between brand and auth) looks unprofessional on small screens.
 * Placing it in the far-right cluster keeps scannable left→right hierarchy:
 *   Brand  |  [desktop links]  |  Health · UserMenu · Hamburger
 */

import Link from "next/link";

const linkCls =
  "px-3 py-1.5 rounded-md text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--surface-2)] transition-colors";

export const NAV_LINKS = [
  { href: "/#features", label: "Features" },
  { href: "/explore",   label: "Explore"  },
  { href: "/chat",      label: "AI Chat"  },
  { href: "/about",     label: "About"    },
  { href: "/docs",      label: "Docs"     },
];

export function NavLinks() {
  return (
    <nav className="hidden md:flex items-center gap-1">
      {NAV_LINKS.map(l => (
        <Link key={l.href} href={l.href} className={linkCls}>
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
