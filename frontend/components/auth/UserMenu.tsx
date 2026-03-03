/**
 * components/auth/UserMenu.tsx
 * =============================
 * "use client" — displays the authenticated user's avatar and email in the
 * header, or a "Sign In" button when the user is not logged in.
 *
 * WHY "use client":
 *   useSession() is a React hook — it can only run in a Client Component.
 *   The Header itself is a Server Component; UserMenu is the client "island"
 *   that accesses session state.
 *
 * DESIGN:
 *   - Authenticated: small avatar circle (initials) + display_name/email + role badge
 *   - Hover: dropdown with "Sign Out" option (matches existing dropdown patterns)
 *   - Unauthenticated: "Sign In" link button styled in gold
 *   - Loading: skeleton shimmer (avoids layout shift during session hydration)
 *
 * WHY use initials avatar (not image):
 *   We don't store profile pictures — keeping the DB simple for the PFE scope.
 *   Initials avatars are generated from display_name or email prefix.
 */

"use client";

import { useState, useRef, useEffect } from "react";
import { useSession, signOut }         from "next-auth/react";
import Link                            from "next/link";
import { motion, AnimatePresence }     from "framer-motion";
import { LogOut, ChevronDown, ShieldCheck, User2 } from "lucide-react";

// ── helpers ───────────────────────────────────────────────────────────────────

function initials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

function avatarColor(email: string): string {
  // Deterministic colour from email hash — consistent across reloads
  let hash = 0;
  for (let i = 0; i < email.length; i++) hash = email.charCodeAt(i) + ((hash << 5) - hash);
  const colours = ["#b45309", "#447a6e", "#5b52a3", "#a05195", "#2f6699", "#6b7280"];
  return colours[Math.abs(hash) % colours.length];
}

const ROLE_LABELS: Record<string, string> = {
  admin:    "Admin",
  curator:  "Curator",
  analyst:  "Analyst",
};

const ROLE_COLOURS: Record<string, string> = {
  admin:   "rgba(239,68,68,0.15)",
  curator: "rgba(212,175,55,0.15)",
  analyst: "rgba(100,116,139,0.2)",
};

const ROLE_TEXT: Record<string, string> = {
  admin:   "#fca5a5",
  curator: "var(--brand-gold)",
  analyst: "var(--text-muted)",
};

// ── component ─────────────────────────────────────────────────────────────────

export function UserMenu() {
  const { data: session, status } = useSession();
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  // ── loading skeleton ───────────────────────────────────────────────────────
  if (status === "loading") {
    return (
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-full animate-pulse" style={{ background: "var(--surface-2)" }} />
        <div className="w-20 h-3 rounded animate-pulse" style={{ background: "var(--surface-2)" }} />
      </div>
    );
  }

  // ── unauthenticated ────────────────────────────────────────────────────────
  if (!session) {
    return (
      <div className="flex items-center gap-2">
        <Link
          href="/login"
          className="px-4 py-1.5 rounded-lg text-sm font-semibold transition-opacity hover:opacity-80"
          style={{ background: "var(--brand-gold)", color: "#0d1520" }}
        >
          Sign In
        </Link>
        <Link
          href="/register"
          className="px-3 py-1.5 rounded-lg text-sm transition-colors"
          style={{ color: "var(--text-secondary)", border: "1px solid var(--border)" }}
        >
          Register
        </Link>
      </div>
    );
  }

  // ── authenticated ──────────────────────────────────────────────────────────
  const user   = session.user;
  const label  = user.display_name || user.name || user.email || "User";
  const role   = (user as { role?: string }).role ?? "analyst";
  const bg     = avatarColor(user.email ?? label);
  const abbrev = initials(label);

  return (
    <div ref={menuRef} className="relative">
      {/* Trigger button */}
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-2 px-2 py-1 rounded-lg transition-colors hover:bg-[var(--surface-2)]"
      >
        {/* Avatar circle */}
        <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0"
             style={{ background: bg, color: "#fff" }}>
          {abbrev}
        </div>

        {/* Name + role — hidden on small screens */}
        <div className="hidden sm:flex flex-col items-start leading-tight">
          <span className="text-sm font-medium truncate max-w-[120px]" style={{ color: "var(--text-primary)" }}>
            {label}
          </span>
          <span className="text-xs px-1.5 py-0 rounded-full"
                style={{ background: ROLE_COLOURS[role], color: ROLE_TEXT[role] }}>
            {ROLE_LABELS[role] ?? role}
          </span>
        </div>

        <ChevronDown
          size={14}
          className="transition-transform"
          style={{ color: "var(--text-muted)", transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        />
      </button>

      {/* Dropdown menu */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, scale: 0.96 }}
            animate={{ opacity: 1, y: 0,  scale: 1 }}
            exit={{    opacity: 0, y: -8, scale: 0.96 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 top-full mt-2 w-52 rounded-xl py-1 z-50 shadow-xl"
            style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
          >
            {/* User info header */}
            <div className="px-4 py-2.5 border-b" style={{ borderColor: "var(--border)" }}>
              <p className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                {label}
              </p>
              <p className="text-xs truncate" style={{ color: "var(--text-muted)" }}>
                {user.email}
              </p>
            </div>

            {/* Profile / admin links */}
            <div className="py-1">
              {(role === "admin" || role === "curator") && (
                <Link
                  href="/admin"
                  onClick={() => setOpen(false)}
                  className="flex items-center gap-2 px-4 py-2 text-sm hover:bg-[var(--surface-2)] transition-colors"
                  style={{ color: role === "admin" ? "#fca5a5" : "var(--brand-gold)" }}
                >
                  <ShieldCheck size={15} />
                  Admin Dashboard
                </Link>
              )}
              <Link
                href="/history"
                onClick={() => setOpen(false)}
                className="flex items-center gap-2 px-4 py-2 text-sm hover:bg-[var(--surface-2)] transition-colors"
                style={{ color: "var(--text-secondary)" }}
              >
                <User2 size={15} />
                My History
              </Link>
            </div>

            {/* Sign out */}
            <div className="py-1 border-t" style={{ borderColor: "var(--border)" }}>
              <button
                onClick={() => { setOpen(false); signOut({ callbackUrl: "/" }); }}
                className="w-full flex items-center gap-2 px-4 py-2 text-sm hover:bg-[var(--surface-2)] transition-colors"
                style={{ color: "var(--text-muted)" }}
              >
                <LogOut size={15} />
                Sign out
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
