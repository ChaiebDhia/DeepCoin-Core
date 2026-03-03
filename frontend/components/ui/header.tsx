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
import { HealthDot }  from "@/components/ui/health-dot";
import { UserMenu }   from "@/components/auth/UserMenu";
import { NavLinks }   from "@/components/ui/NavLinks";

export function Header() {
  return (
    <header
      className="sticky top-0 z-50 w-full border-b border-[var(--border)]"
      style={{ backgroundColor: "var(--surface-1)" }}
    >
      <div className="mx-auto max-w-6xl px-5 h-14 flex items-center justify-between gap-4">
        {/* Brand */}
        <Link
          href="/"
          className="flex items-center gap-2 text-[var(--text-primary)] hover:text-white transition-colors"
        >
          <Coins size={22} className="text-[var(--brand-gold)]" />
          <span className="font-bold tracking-tight text-base">
            Deep<span style={{ color: "var(--brand-gold)" }}>Coin</span>
          </span>
        </Link>

        {/* Nav — auth-aware client island */}
        <NavLinks />

        {/* Right cluster: health dot + auth menu */}
        <div className="flex items-center gap-3">
          {/* Health indicator — left of the auth buttons */}
          <HealthDot />

          {/* Auth: avatar dropdown when logged in, Sign In / Register otherwise */}
          <UserMenu />
        </div>
      </div>
    </header>
  );
}
