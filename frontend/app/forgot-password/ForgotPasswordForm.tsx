/**
 * app/forgot-password/ForgotPasswordForm.tsx
 * ============================================
 * "use client" — interactive form for the forgot-password flow.
 *
 * WHAT it does:
 *   Submits the user's email to POST /auth/forgot-password.
 *   On success (always 200), replaces the form with a branded confirmation.
 *   On network error, shows a dismissable error message.
 *
 * WHY no success/failure distinction based on whether the email exists:
 *   The server deliberately returns 200 regardless — this is a SECURITY
 *   requirement.  We match that behaviour client-side: the user always
 *   sees "check your inbox" so attackers cannot probe for account existence.
 */

"use client";

import { useState, FormEvent } from "react";
import Link                    from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Mail, Coins, AlertCircle, Loader2, CheckCircle2 } from "lucide-react";
import { forgotPassword } from "@/lib/api";

export default function ForgotPasswordForm() {
  const [email,   setEmail]   = useState("");
  const [loading, setLoading] = useState(false);
  const [sent,    setSent]    = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await forgotPassword(email);
      setSent(true);
    } catch {
      setError("Something went wrong. Please try again or contact support.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full max-w-sm mx-auto"
    >
      {/* Brand mark */}
      <div className="text-center mb-8">
        <div
          className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-4"
          style={{
            background: "linear-gradient(135deg, rgba(212,175,55,0.15) 0%, rgba(212,175,55,0.05) 100%)",
            border:     "1px solid rgba(212,175,55,0.3)",
          }}
        >
          <Coins size={28} style={{ color: "var(--brand-gold)" }} />
        </div>
        <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
          Reset your password
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
          Enter your email and we&rsquo;ll send you a reset link
        </p>
      </div>

      <div
        className="rounded-2xl p-6"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <AnimatePresence mode="wait">

          {/* ── Success state ────────────────────────────────────────── */}
          {sent ? (
            <motion.div
              key="success"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center space-y-3 py-4"
            >
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full"
                   style={{ background: "rgba(16,185,129,0.12)" }}>
                <CheckCircle2 size={24} style={{ color: "#10b981" }} />
              </div>
              <div>
                <p className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>
                  Check your inbox
                </p>
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                  If <strong>{email}</strong> is associated with an account,
                  a password reset link has been sent. It expires in 1 hour.
                </p>
              </div>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                Didn&rsquo;t receive it?{" "}
                <button
                  onClick={() => { setSent(false); setEmail(""); }}
                  className="underline hover:opacity-80"
                  style={{ color: "var(--brand-gold)" }}
                >
                  Try a different email
                </button>
              </p>
            </motion.div>
          ) : (

          /* ── Form state ─────────────────────────────────────────── */
            <motion.form
              key="form"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              onSubmit={handleSubmit}
              className="space-y-4"
            >
              {/* Error banner */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="flex items-start gap-2 text-sm px-3 py-2.5 rounded-lg"
                  style={{
                    background: "rgba(239,68,68,0.1)",
                    border: "1px solid rgba(239,68,68,0.3)",
                    color: "#fca5a5",
                  }}
                >
                  <AlertCircle size={16} className="mt-0.5 shrink-0" />
                  <span>{error}</span>
                </motion.div>
              )}

              {/* Email field */}
              <div className="space-y-1.5">
                <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                  Email address
                </label>
                <div className="relative">
                  <Mail
                    size={16}
                    className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                    style={{ color: "var(--text-muted)" }}
                  />
                  <input
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    required
                    autoComplete="email"
                    autoFocus
                    placeholder="you@example.com"
                    className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm outline-none transition-colors"
                    style={{
                      background: "var(--surface-2)",
                      border:     "1px solid var(--border)",
                      color:      "var(--text-primary)",
                    }}
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
                  ? <><Loader2 size={16} className="animate-spin" /> Sending link…</>
                  : "Send reset link"}
              </button>
            </motion.form>
          )}
        </AnimatePresence>
      </div>

      {/* Footer links */}
      <p className="text-center text-sm mt-4" style={{ color: "var(--text-muted)" }}>
        Remembered it?{" "}
        <Link href="/login" className="font-medium hover:underline" style={{ color: "var(--brand-gold)" }}>
          Sign in
        </Link>
      </p>
    </motion.div>
  );
}
