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
import { useTranslations } from "next-intl";
import Link                    from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Mail, Lock, User, Coins, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { signIn } from "next-auth/react";

// FastAPI base URL — NEXT_PUBLIC so the browser can call it directly
const API_BASE = process.env.NEXT_PUBLIC_CLASSIFY_URL ?? "http://127.0.0.1:8000";

export function RegisterForm() {
  const t = useTranslations("AuthForms");
  const [displayName,      setDisplayName]      = useState("");
  const [email,            setEmail]            = useState("");
  const [password,         setPassword]         = useState("");
  const [confirmPassword,  setConfirmPassword]  = useState("");
  const [error,            setError]            = useState<string | null>(null);
  const [success,          setSuccess]          = useState(false);
  const [successMsg,       setSuccessMsg]       = useState<string | null>(null);
  const [loading,          setLoading]          = useState(false);
  const [googleLoading,    setGoogleLoading]    = useState(false);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError(t("err_passwords_match"));
      return;
    }
    if (password.length < 8) {
      setError(t("err_password_length"));
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
        setError(t("err_register_failed"));
      }
    } catch {
      setError(t("err_server_unreachable"));
    } finally {
      setLoading(false);
    }
  }

  async function handleGoogleSignIn() {
    setError(null);
    setGoogleLoading(true);
    try {
      await signIn("google", { callbackUrl: "/dashboard" });
    } catch {
      setError(t("err_google_signin"));
    } finally {
      setGoogleLoading(false);
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
          {t("account_created")}
        </h2>
        <p className="text-sm mb-6" style={{ color: "var(--text-muted)" }}>
          {successMsg ?? (
            <>
              {t("verification_sent_prefix")} <strong style={{ color: "var(--text-secondary)" }}>{email}</strong>.
              {t("verification_sent_suffix")}
            </>
          )}
        </p>
        <Link
          href="/login"
          className="inline-block px-6 py-2.5 rounded-lg text-sm font-semibold"
          style={{ background: "var(--brand-gold)", color: "#0d1520" }}
        >
          {t("go_to_signin")}
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
        <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>{t("create_account_title")}</h1>
        <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
          {t("join_deepcoin_subtitle")} — {t("free_to_get_started")}
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
              {t("display_name_label")} <span style={{ color: "var(--text-muted)" }}>({t("optional")})</span>
            </label>
            <div className="relative">
              <User size={16} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                   style={{ color: "var(--text-muted)" }} />
              <input
                type="text"
                value={displayName}
                onChange={e => setDisplayName(e.target.value)}
                autoComplete="name"
                placeholder={t("display_name_placeholder")}
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
              {t("email_address")}
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
                placeholder={t("email_placeholder")}
                className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm outline-none transition-colors"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                onFocus={e => (e.target.style.borderColor = "var(--brand-gold)")}
                onBlur={e  => (e.target.style.borderColor = "var(--border)")}
              />
            </div>
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>{t("password_label")}</label>
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
                placeholder={t("password_placeholder")}
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
              {t("confirm_password_label")}
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
                placeholder={t("password_placeholder")}
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
            disabled={loading || googleLoading}
            className="w-full py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 transition-opacity disabled:opacity-60"
            style={{ background: "var(--brand-gold)", color: "#0d1520" }}
          >
            {loading
              ? <><Loader2 size={16} className="animate-spin" /> {t("creating_account")}</>
              : t("create_account")}
          </button>

          <div className="relative py-1">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t" style={{ borderColor: "var(--border)" }} />
            </div>
            <div className="relative flex justify-center text-xs uppercase tracking-wide">
              <span className="px-2" style={{ background: "var(--surface-1)", color: "var(--text-muted)" }}>{t("or_text")}</span>
            </div>
          </div>

          <button
            type="button"
            onClick={handleGoogleSignIn}
            disabled={loading || googleLoading}
            className="w-full py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 transition-opacity disabled:opacity-60 hover:opacity-85"
            style={{ background: "var(--surface-2)", color: "var(--text-primary)", border: "1px solid var(--border)" }}
          >
            {googleLoading
              ? <><Loader2 size={16} className="animate-spin" /> {t("redirect_google")}</>
              : <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
                  <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                  <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
                  <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                </svg>
                {t("continue_with_google")}
              </>
            }
          </button>

        </form>
      </div>

      {/* Footer link */}
      <p className="text-center text-sm mt-4" style={{ color: "var(--text-muted)" }}>
        {t("already_have_account")} {" "}
        <Link href="/login" className="font-medium hover:underline" style={{ color: "var(--brand-gold)" }}>{t("sign_in_link")}</Link>
      </p>
    </motion.div>
  );
}
