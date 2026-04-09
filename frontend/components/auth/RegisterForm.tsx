/**
 * components/auth/RegisterForm.tsx
 * =================================
 * "use client" — registration form.
 *
 * FLOW:
 *   1. User fills display_name (optional), email, password, confirm_password
 *   2. Form POSTs directly to FastAPI POST /auth/register
 *      WHY direct to FastAPI (not through signIn):
 *        Registration creates the account + triggers verification email.
 *        signIn() is for authenticating an EXISTING account.
 *        After registration, the user must verify their email before they can
 *        sign in, so auto-login after register would fail with status=pending.
 *   3. On success → show "Check your email" message (no auto-login)
 *   4. On duplicate email → show "Email already registered"
 *
 * PASSWORD RULES (enforced client-side + server-side):
 *   - Minimum 8 characters
 *   - Confirm password must match
 *   FastAPI enforces additional strength requirements server-side.
 */

"use client";

import { useState, FormEvent } from "react";
import Link                    from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Mail, Lock, User, Coins, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";

// FastAPI base URL — NEXT_PUBLIC so the browser can call it directly
const API_BASE = process.env.NEXT_PUBLIC_CLASSIFY_URL ?? "http://127.0.0.1:8000";

export function RegisterForm() {
  const [displayName,      setDisplayName]      = useState("");
  const [email,            setEmail]            = useState("");
  const [password,         setPassword]         = useState("");
  const [confirmPassword,  setConfirmPassword]  = useState("");
  const [error,            setError]            = useState<string | null>(null);
  const [success,          setSuccess]          = useState(false);
  const [successMsg,       setSuccessMsg]       = useState<string | null>(null);
  const [loading,          setLoading]          = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password,
          display_name: displayName.trim() || null,
        }),
      });

      if (res.ok) {
        const data = await res.json().catch(() => ({})) as { message?: string };
        setSuccessMsg(data.message ?? null);
        setSuccess(true);
        return;
      }

      const data = await res.json().catch(() => ({})) as { detail?: string | { msg: string }[] };
      if (Array.isArray(data.detail)) {
        setError(data.detail.map(d => d.msg).join(" · "));
      } else if (typeof data.detail === "string") {
        // FastAPI returns "Email already registered" as a plain string
        setError(data.detail);
      } else {
        setError("Registration failed. Please try again.");
      }
    } catch {
      setError("Could not reach the server. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  // ── success state ──────────────────────────────────────────────────────────

  if (success) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-sm mx-auto text-center"
      >
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-full mb-4"
             style={{ background: "var(--surface-1)", border: "1px solid var(--surface-1)" }}>
          <CheckCircle2 size={28} style={{ color: "#22c55e" }} />
        </div>
        <h2 className="text-xl font-bold mb-2" style={{ color: "var(--text-primary)" }}>
          Account created!
        </h2>
        <p className="text-sm mb-6" style={{ color: "var(--text-muted)" }}>
          {successMsg ?? (
            <>
              We sent a verification link to <strong style={{ color: "var(--text-secondary)" }}>{email}</strong>.
              Click the link to activate your account, then sign in.
            </>
          )}
        </p>
        <Link
          href="/login"
          className="inline-block px-6 py-2.5 rounded-lg text-sm font-semibold"
          style={{ background: "var(--brand-gold)", color: "#0d1520" }}
        >
          Go to Sign In
        </Link>
      </motion.div>
    );
  }

  // ── form ───────────────────────────────────────────────────────────────────

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full max-w-sm mx-auto"
    >
      {/* Brand mark */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-4"
             style={{ background: "linear-gradient(135deg, var(--surface-1) 0%, var(--surface-1) 100%)", border: "1px solid var(--surface-1)" }}>
          <Coins size={28} style={{ color: "var(--brand-gold)" }} />
        </div>
        <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
          Create account
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
          Join DeepCoin — free to get started
        </p>
      </div>

      {/* Card */}
      <div className="rounded-2xl p-6" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
        <form onSubmit={handleSubmit} className="space-y-4">

          {/* Error banner */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="flex items-start gap-2 text-sm px-3 py-2.5 rounded-lg"
                style={{ background: "var(--surface-1)", border: "1px solid var(--surface-1)", color: "#fca5a5" }}
              >
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Display name (optional) */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
              Display name <span style={{ color: "var(--text-muted)" }}>(optional)</span>
            </label>
            <div className="relative">
              <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                   style={{ color: "var(--text-muted)" }} />
              <input
                type="text"
                value={displayName}
                onChange={e => setDisplayName(e.target.value)}
                autoComplete="name"
                placeholder="Dr. Ahmed Chaieb"
                className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm outline-none transition-colors"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                onFocus={e => (e.target.style.borderColor = "var(--brand-gold)")}
                onBlur={e  => (e.target.style.borderColor = "var(--border)")}
              />
            </div>
          </div>

          {/* Email */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
              Email address
            </label>
            <div className="relative">
              <Mail size={16} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                   style={{ color: "var(--text-muted)" }} />
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
                placeholder="you@example.com"
                className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm outline-none transition-colors"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                onFocus={e => (e.target.style.borderColor = "var(--brand-gold)")}
                onBlur={e  => (e.target.style.borderColor = "var(--border)")}
              />
            </div>
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
              Password
            </label>
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                   style={{ color: "var(--text-muted)" }} />
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                minLength={8}
                autoComplete="new-password"
                placeholder="Minimum 8 characters"
                className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm outline-none transition-colors"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                onFocus={e => (e.target.style.borderColor = "var(--brand-gold)")}
                onBlur={e  => (e.target.style.borderColor = "var(--border)")}
              />
            </div>
          </div>

          {/* Confirm password */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
              Confirm password
            </label>
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                   style={{ color: "var(--text-muted)" }} />
              <input
                type="password"
                value={confirmPassword}
                onChange={e => setConfirmPassword(e.target.value)}
                required
                autoComplete="new-password"
                placeholder="••••••••"
                className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm outline-none transition-colors"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                onFocus={e => (e.target.style.borderColor = "var(--brand-gold)")}
                onBlur={e  => (e.target.style.borderColor = "var(--border)")}
              />
            </div>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 transition-opacity disabled:opacity-60"
            style={{ background: "var(--brand-gold)", color: "#0d1520" }}
          >
            {loading
              ? <><Loader2 size={16} className="animate-spin" /> Creating account…</>
              : "Create account"}
          </button>

        </form>
      </div>

      {/* Footer link */}
      <p className="text-center text-sm mt-4" style={{ color: "var(--text-muted)" }}>
        Already have an account?{" "}
        <Link href="/login" className="font-medium hover:underline" style={{ color: "var(--brand-gold)" }}>
          Sign in
        </Link>
      </p>
    </motion.div>
  );
}
