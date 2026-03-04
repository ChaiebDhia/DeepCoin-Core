"use client";

/**
 * components/ui/MobileNav.tsx
 * ============================
 * Mobile-only hamburger menu rendered inside the header's RIGHT cluster.
 *
 * WHY placed on the right:
 *   The conventional position for a hamburger on mobile is the far right
 *   (or far left, but right is more common in SaaS / enterprise apps).
 *   Placing it in the RIGHT cluster means it sits adjacent to the UserMenu
 *   and HealthDot — so all interactive controls are grouped at one end.
 *   The brand stays anchored on the left, exactly as on desktop.
 *
 * WHAT it renders:
 *   - A compact square hamburger button (md:hidden)
 *   - A slide-down card with all nav links + a divider + auth shortcuts
 *   - The card is 220px wide, right-aligned (never overflows the viewport)
 *
 * HOW close works:
 *   - Click outside the ref'd container
 *   - Click any link (onClick closes)
 *   - Press Escape
 */

import { useState, useEffect, useRef } from "react";
import Link                             from "next/link";
import { usePathname }                  from "next/navigation";
import { Menu, X, Cpu, BookOpen, MessageSquare, FlaskConical, Globe, FileText } from "lucide-react";
import { useSession }                   from "next-auth/react";
import { NAV_LINKS }                    from "@/components/ui/NavLinks";

const ICON_MAP: Record<string, React.ReactNode> = {
  "/#features": <Cpu size={14} />,
  "/explore":   <Globe size={14} />,
  "/chat":      <MessageSquare size={14} />,
  "/about":     <FlaskConical size={14} />,
  "/docs":      <FileText size={14} />,
};

export function MobileNav() {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const { data: session } = useSession();
  const pathname = usePathname();

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function onOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, [open]);

  return (
    <div className="md:hidden relative" ref={menuRef}>
      {/* Hamburger trigger */}
      <button
        onClick={() => setOpen(v => !v)}
        aria-label={open ? "Close menu" : "Open menu"}
        aria-expanded={open}
        className={`flex items-center justify-center w-9 h-9 rounded-lg transition-all
          ${open
            ? "bg-[var(--surface-2)] text-white"
            : "text-[var(--text-secondary)] hover:bg-[var(--surface-2)] hover:text-white"
          }`}
      >
        {open ? <X size={18} /> : <Menu size={18} />}
      </button>

      {/* Dropdown card */}
      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full mt-2 w-56 rounded-2xl overflow-hidden select-none"
          style={{
            background: "var(--surface-1)",
            border:     "1px solid var(--border)",
            boxShadow:  "0 20px 56px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.04)",
            zIndex:     60,
          }}
        >
          {/* Nav links */}
          <div className="py-1.5">
            <p
              className="px-4 py-1.5 text-[10px] font-semibold uppercase tracking-widest"
              style={{ color: "var(--text-muted)" }}
            >
              Navigation
            </p>
            {NAV_LINKS.map(l => {
              const active = !l.href.startsWith("/#") && (
                pathname === l.href || pathname.startsWith(l.href + "/")
              );
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  role="menuitem"
                  onClick={() => setOpen(false)}
                  className="flex items-center gap-3 px-4 py-2.5 text-sm transition-colors hover:bg-[var(--surface-2)]"
                  style={{
                    color:      active ? "var(--brand-gold, #d4a853)" : "var(--text-secondary)",
                    fontWeight: active ? 600 : undefined,
                  }}
                >
                  <span style={{ color: active ? "var(--brand-gold, #d4a853)" : "var(--text-muted)" }}>
                    {ICON_MAP[l.href]}
                  </span>
                  {l.label}
                  {active && (
                    <span
                      className="ml-auto w-1.5 h-1.5 rounded-full"
                      style={{ background: "var(--brand-gold, #d4a853)" }}
                    />
                  )}
                </Link>
              );
            })}
          </div>

          {/* Divider + auth shortcuts */}
          <div style={{ borderTop: "1px solid var(--border)" }}>
            <div className="py-1.5">
              <p
                className="px-4 py-1.5 text-[10px] font-semibold uppercase tracking-widest"
                style={{ color: "var(--text-muted)" }}
              >
                {session ? "My Account" : "Get Started"}
              </p>
              {session ? (
                <>
                  <Link
                    href="/analyse"
                    role="menuitem"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-3 px-4 py-2.5 text-sm transition-colors hover:bg-[var(--surface-2)]"
                    style={{ color: "var(--brand-gold)" }}
                  >
                    <Cpu size={14} />
                    Analyse a Coin
                  </Link>
                  <Link
                    href="/history"
                    role="menuitem"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-3 px-4 py-2.5 text-sm transition-colors hover:bg-[var(--surface-2)]"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    <BookOpen size={14} />
                    My History
                  </Link>
                </>
              ) : (
                <>
                  <Link
                    href="/login"
                    role="menuitem"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-3 px-4 py-2.5 text-sm transition-colors hover:bg-[var(--surface-2)]"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Sign In
                  </Link>
                  <Link
                    href="/register"
                    role="menuitem"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-3 px-4 py-2.5 text-sm font-semibold transition-colors hover:bg-[var(--surface-2)]"
                    style={{ color: "var(--brand-gold)" }}
                  >
                    Create Account
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
