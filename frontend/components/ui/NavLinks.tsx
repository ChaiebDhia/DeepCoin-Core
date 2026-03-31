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

import Link        from "next/link";
import { usePathname } from "next/navigation";

/**
 * Returns true when the nav link should be highlighted.
 * - Hash links (/#features) are never active (they anchor to the homepage).
 * - All other links are active when the pathname starts with the href
 *   (e.g. /chat → active, /chat/session-123 → also active).
 */
function isActive(href: string, pathname: string): boolean {
  if (href.startsWith("/#")) return false;
  // Exact match for root; prefix match for all others.
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(href + "/");
}

const baseCls =
  "px-3 py-1.5 rounded-md text-sm transition-all relative";
const inactiveCls =
  "text-[var(--text-secondary)] hover:text-white hover:bg-[var(--surface-2)]";
/** Active: white bold text + subtle surface background so it's visually
 *  unmistakable even if the 2px gold underline is missed at a glance. */
const activeCls =
  "text-white font-semibold bg-[var(--surface-2)]";

export const NAV_LINKS = [
  { href: "/",          label: "Home"     },
  { href: "/explore",   label: "Explore"  },
  { href: "/chat",      label: "AI Chat"  },
  { href: "/about",     label: "About"    },
];

export function NavLinks() {
  const pathname = usePathname();

  return (
    <nav className="hidden md:flex items-center gap-1">
      {NAV_LINKS.map(l => {
        const active = isActive(l.href, pathname);
        return (
          <Link
            key={l.href}
            href={l.href}
            className={`${baseCls} ${active ? activeCls : inactiveCls}`}
          >
            {l.label}
            {/* Gold underline bar for active page */}
            {active && (
              <span
                className="absolute bottom-0 left-3 right-3 h-[2px] rounded-full"
                style={{ background: "var(--brand-gold, #d4a853)" }}
              />
            )}
          </Link>
        );
      })}
    </nav>
  );
}
