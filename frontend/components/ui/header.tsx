/**
 * components/ui/header.tsx
 * ========================
 * Top navigation bar — appears on every page.
 * Shows:
 *   - DeepCoin brand mark (links to home)
 *   - Navigation links (home, history)
 *   - Backend health indicator (dots: green=ok, amber=degraded, red=down)
 *
 * WHY server-rendered here but health is client-fetched:
 *   The nav structure is static — it can be server-rendered.
 *   The health dot needs a live API call — it's in a child client component.
 */

import Link           from "next/link";
import { Coins }      from "lucide-react";
import { HealthDot }  from "@/components/ui/health-dot";
import { UserMenu }   from "@/components/auth/UserMenu";

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

        {/* Nav */}
        <nav className="flex items-center gap-1">
          <Link
            href="/#features"
            className="px-3 py-1.5 rounded-md text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--surface-2)] transition-colors"
          >
            Features
          </Link>
          <Link
            href="/#analyse"
            className="px-3 py-1.5 rounded-md text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--surface-2)] transition-colors"
          >
            Analyse
          </Link>
          <Link
            href="/history"
            className="px-3 py-1.5 rounded-md text-sm text-[var(--text-secondary)] hover:text-white hover:bg-[var(--surface-2)] transition-colors"
          >
            History
          </Link>
        </nav>

        {/* Auth: user menu when logged in, Sign In link otherwise */}
        <UserMenu />

        {/* Health indicator */}
        <HealthDot />
      </div>
    </header>
  );
}
