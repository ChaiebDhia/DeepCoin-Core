/**
 * components/ui/header.tsx
 * ========================
 * Top navigation bar — appears on every page.
 * Shows:
 *   - DeepCoin brand mark (links to home)
 *   - Navigation links via NavLinks (auth-aware client component)
 *   - Backend health indicator (live-polling green/amber/red dot)
 *   - Auth area: UserMenu (avatar + dropdown) or Sign In / Register
 *
 * WHY HealthDot is LEFT of UserMenu:
 *   Auth controls (avatar, Sign In, Register) are fixed at the far right —
 *   the conventional position users reach for. The health dot is a
 *   secondary indicator, naturally placed just left of the auth cluster.
 *
 * WHY NavLinks is a client component but header.tsx is a server component:
 *   The "Analyse" link must be auth-aware (redirect to login when signed out).
 *   useSession() requires "use client". Extracting just the nav keeps the
 *   rest of the header server-rendered.
 */

import Link           from "next/link";
import { Coins }      from "lucide-react";
import { UserMenu }    from "@/components/auth/UserMenu";
import { NavLinks }    from "@/components/ui/NavLinks";
import { MobileNav }   from "@/components/ui/MobileNav";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { LanguageToggle } from "@/components/ui/LanguageToggle";

export function Header() {
  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 w-full border-b backdrop-blur-xl transition-all duration-300"
      style={{ backgroundColor: "var(--nav-bg-glass)", borderColor: "var(--border)", color: "var(--nav-text)" }}
    >
      <div className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4 sm:gap-6">
        {/* Brand */}
        <Link
          href="/"
          className="flex items-center gap-2 text-[var(--nav-text)] hover:text-[#fbbf24] transition-colors flex-shrink-0"
        >
          <Coins size={24} className="text-[var(--brand-gold)]" />
          <span className="font-bold tracking-tight text-lg hidden sm:inline">
            Deep<span style={{ color: "var(--brand-gold)" }}>Coin</span>
          </span>
        </Link>

        {/* Nav — auth-aware client island, centered */}
        <div className="flex-1 flex justify-center">
          <NavLinks />
        </div>

        {/* Right cluster: auth menu + mobile hamburger */}
        <div className="flex items-center gap-4 flex-shrink-0">
          <div className="hidden lg:flex items-center gap-2">
            <LanguageToggle />
            <ThemeToggle />
          </div>

          {/* Auth: avatar dropdown when logged in, Sign In otherwise */}
          <UserMenu />

          {/* Mobile hamburger — far right, only renders below lg */}
          <MobileNav />
        </div>
      </div>
    </header>
  );
}
