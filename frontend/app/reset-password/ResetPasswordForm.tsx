/**
 * app/reset-password/ResetPasswordForm.tsx
 * ==========================================
 * "use client" — interactive form to set a new password.
 *
 * WHAT it does:
 *   1. Reads the ?token= query parameter from the URL.
 *   2. Shows a new-password / confirm-password form.
 *   3. POSTs { token, new_password } to /auth/reset-password.
 *   4. On success: shows branded confirmation + redirects to /login after 3 s.
 *   5. On 400 (invalid/expired token): shows actionable error with link to
 *      request another reset.
 *
 * WHY redirect instead of instant navigation:
 *   A brief success message gives the user visual confirmation before the page
 *   changes.  Three seconds is enough to read the message without feeling
 *   abandoned by an instant redirect.
 */

"use client";

import { useState, FormEvent, useEffect } from "react";
import { useSearchParams, useRouter }      from "next/navigation";
import Link                               from "next/link";
import { motion, AnimatePresence }        from "framer-motion";
import { Lock, Coins, AlertCircle, Loader2, CheckCircle2, Eye, EyeOff } from "lucide-react";
import { resetPassword } from "@/lib/api";

export default function ResetPasswordForm() {
  const searchParams = useSearchParams();
  const router       = useRouter();
  const token        = searchParams.get("token") ?? "";

  const [password,  setPassword]  = useState("");
  const [confirm,   setConfirm]   = useState("");
  const [showPwd,   setShowPwd]   = useState(false);
  const [loading,   setLoading]   = useState(false);
  const [done,      setDone]      = useState(false);
  const [error,     setError]     = useState<string | null>(null);

  // Auto-redirect to /login 3 s after success
  useEffect(() => {
    if (!done) return;
    const t = setTimeout(() => router.push("/login"), 3000);
    return () => clearTimeout(t);
  }, [done, router]);

  // Missing token means the user navigated here without a link
  if (!token) {
    return (
      <div className="w-full max-w-sm mx-auto text-center space-y-4 py-16">
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          No reset token found. Please{" "}
          <Link href="/forgot-password" className="underline" style={{ color: "var(--brand-gold)" }}>
            request a new password reset
          </Link>
          .
        </p>
      </div>
    );
  }

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);

    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    setLoading(true);
    try {
      await resetPassword(token, password);
      setDone(true);
    } catch (err: unknown) {
      // FastAPI returns 400 with a detail string for invalid/expired tokens
      const detail =
        (err as { response?: { data?: { detail?: string } } })
          ?.response?.data?.detail;
      setError(detail ?? "Reset failed. The link may have expired.");
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
          Set a new password
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
          Choose something strong — at least 8 characters
        </p>
      </div>

      <div
        className="rounded-2xl p-6"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <AnimatePresence mode="wait">

          {/* ── Success state ────────────────────────────────────────── */}
          {done ? (
            <motion.div
              key="success"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center space-y-3 py-4"
            >
              <div
                className="inline-flex items-center justify-center w-12 h-12 rounded-full"
                style={{ background: "rgba(16,185,129,0.12)" }}
              >
                <CheckCircle2 size={24} style={{ color: "#10b981" }} />
              </div>
              <div>
                <p className="font-semibold text-sm" style={{ color: "var(--text-primary)" }}>
                  Password updated successfully!
                </p>
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                  Redirecting you to sign in&hellip;
                </p>
              </div>
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
                  className="flex flex-col gap-1.5 text-sm px-3 py-2.5 rounded-lg"
                  style={{
                    background: "rgba(239,68,68,0.1)",
                    border: "1px solid rgba(239,68,68,0.3)",
                    color: "#fca5a5",
                  }}
                >
                  <div className="flex items-start gap-2">
                    <AlertCircle size={16} className="mt-0.5 shrink-0" />
                    <span>{error}</span>
                  </div>
                  {error.includes("expired") && (
                    <Link
                      href="/forgot-password"
                      className="ml-6 text-xs underline underline-offset-2 hover:opacity-80"
                      style={{ color: "#fcd34d" }}
                    >
                      Request a new reset link →
                    </Link>
                  )}
                </motion.div>
              )}

              {/* New password */}
              <div className="space-y-1.5">
                <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                  New password
                </label>
                <div className="relative">
                  <Lock
                    size={16}
                    className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                    style={{ color: "var(--text-muted)" }}
                  />
                  <input
                    type={showPwd ? "text" : "password"}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                    autoComplete="new-password"
                    autoFocus
                    placeholder="Minimum 8 characters"
                    className="w-full pl-9 pr-10 py-2.5 rounded-lg text-sm outline-none transition-colors"
                    style={{
                      background: "var(--surface-2)",
                      border:     "1px solid var(--border)",
                      color:      "var(--text-primary)",
                    }}
                    onFocus={e => (e.target.style.borderColor = "var(--brand-gold)")}
                    onBlur={e  => (e.target.style.borderColor = "var(--border)")}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPwd(v => !v)}
                    className="absolute right-3 top-1/2 -translate-y-1/2"
                    style={{ color: "var(--text-muted)" }}
                    tabIndex={-1}
                  >
                    {showPwd ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
              </div>

              {/* Confirm password */}
              <div className="space-y-1.5">
                <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                  Confirm new password
                </label>
                <div className="relative">
                  <Lock
                    size={16}
                    className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                    style={{ color: "var(--text-muted)" }}
                  />
                  <input
                    type={showPwd ? "text" : "password"}
                    value={confirm}
                    onChange={e => setConfirm(e.target.value)}
                    required
                    autoComplete="new-password"
                    placeholder="Repeat password"
                    className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm outline-none transition-colors"
                    style={{
                      background: "var(--surface-2)",
                      border:     `1px solid ${confirm && confirm !== password ? "rgba(239,68,68,0.5)" : "var(--border)"}`,
                      color:      "var(--text-primary)",
                    }}
                    onFocus={e => (e.target.style.borderColor = "var(--brand-gold)")}
                    onBlur={e  => (e.target.style.borderColor =
                      confirm && confirm !== password ? "rgba(239,68,68,0.5)" : "var(--border)"
                    )}
                  />
                </div>
                {confirm && confirm !== password && (
                  <p className="text-xs" style={{ color: "#fca5a5" }}>Passwords don&rsquo;t match</p>
                )}
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={loading || !password || !confirm}
                className="w-full py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 transition-opacity disabled:opacity-60"
                style={{ background: "var(--brand-gold)", color: "#0d1520" }}
              >
                {loading
                  ? <><Loader2 size={16} className="animate-spin" /> Updating password…</>
                  : "Update password"}
              </button>
            </motion.form>
          )}
        </AnimatePresence>
      </div>

      <p className="text-center text-sm mt-4" style={{ color: "var(--text-muted)" }}>
        <Link href="/login" className="font-medium hover:underline" style={{ color: "var(--brand-gold)" }}>
          Back to sign in
        </Link>
      </p>
    </motion.div>
  );
}
