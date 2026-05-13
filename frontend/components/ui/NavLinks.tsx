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
import { useTranslations } from "next-intl";
import { Home, Globe, MessageSquare, Info } from "lucide-react";

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
  "flex items-center gap-1.5 px-3 py-2 sm:px-4 rounded-lg text-sm font-medium transition-all duration-200";
const inactiveCls =
  "text-[var(--nav-text)] opacity-70 hover:opacity-100 hover:bg-white/10";
/** Active: highly visible bold text + subtle surface background + gold bottom border */
const activeCls =
  "text-white font-bold bg-white/10 border-b-2 border-[#d4a853]";

export const NAV_LINKS = [
  { href: "/",          labelKey: "home", icon: Home },
  { href: "/explore",   labelKey: "explore", icon: Globe },
  { href: "/chat",      labelKey: "chat", icon: MessageSquare },
  { href: "/about",     labelKey: "about", icon: Info },
];

export function NavLinks() {
  const t = useTranslations("Navbar");
  const pathname = usePathname();

  return (
    <nav className="hidden lg:flex items-center gap-1">
      {NAV_LINKS.map(l => {
        const active = isActive(l.href, pathname);
        return (
          <Link
            key={l.href}
            href={l.href}
            className={`${baseCls} ${active ? activeCls : inactiveCls}`}
          >
            {l.icon && <l.icon size={16} />}
            <span>{t(l.labelKey)}</span>
          </Link>
        );
      })}
    </nav>
  );
}
