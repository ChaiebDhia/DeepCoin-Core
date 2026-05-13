/**
 * components/auth/LoginForm.tsx
 * ==============================
 * "use client" — interactive login form component.
 *
 * DESIGN:
 *   - Matches the existing DeepCoin dark navy aesthetic
 *   - Email + password fields with animated focus states
 *   - Loading spinner during sign-in (pipeline can take a moment)
 *   - Error display for bad credentials / account suspended
 *   - Redirects to callbackUrl (or /) on success
 *
 * HOW authentication works:
 *   signIn("credentials", { email, password, redirect: false })
 *     → calls auth.config.ts authorize()
 *     → authorize() POSTs to FastAPI /auth/login
 *     → on success, next-auth creates a signed httpOnly cookie
 *     → we redirect to callbackUrl (the page the user was trying to reach)
 *
 * WHY redirect: false:
 *   We want to handle the redirect manually so we can show an error message
 *   on the same page instead of navigating away. If redirect:true, next-auth
 *   would redirect to /api/auth/error on failure.
 */

"use client";

import { useState, FormEvent } from "react";
import { useTranslations } from "next-intl";
import { signIn }              from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";
import Link                    from "next/link";
import { motion }              from "framer-motion";
import { Mail, Lock, Coins, AlertCircle, Loader2, CheckCircle } from "lucide-react";
import { resendVerification }  from "@/lib/api";

export function LoginForm() {
  const t = useTranslations("AuthForms");
  const router       = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl  = searchParams.get("callbackUrl") ?? "/";
  const googleEnabled = process.env.NEXT_PUBLIC_GOOGLE_AUTH_ENABLED === "1";

  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [error,    setError]    = useState<string | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  // Resend verification state — shown when a pending account tries to sign in
  const [resendSent,    setResendSent]    = useState(false);
  const [resendLoading, setResendLoading] = useState(false);

  // ── friendly error messages ────────────────────────────────────────────────

  const ERROR_MESSAGES: Record<string, string> = {
    CredentialsSignin:   t("err_invalid_credentials"),
    CallbackRouteError:  t("err_verify_email"),
    Default:             t("err_default"),
  };

  // Is the current error a "verify your email" failure?
  const isPendingError = error?.includes("verify your email");

  /**
   * Resend verification email for a pending account.
   *
   * WHY inline (not a separate page):
   *   The user's email is already in the form — no need to navigate away.
   *   We call POST /auth/resend-verification (always returns 200) so there
   *   is no need to handle 404/403 edge cases on the client.
   */
  async function handleResend() {
    if (!email || resendLoading) return;
    setResendLoading(true);
    try {
      await resendVerification(email);
      setResendSent(true);
    } finally {
      setResendLoading(false);
    }
  }

  // ── submit handler ─────────────────────────────────────────────────────────

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    if (!email || !password) {
      setError(t("err_fill_fields"));
      return;
    }
    setLoading(true);

    try {
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      });

      if (!result) {
        setError(ERROR_MESSAGES.Default);
        return;
      }

      if (result.error) {
        setError(ERROR_MESSAGES[result.error] ?? ERROR_MESSAGES.Default);
        return;
      }

      // ✅ Success — navigate to the page the user tried to reach
      router.push(callbackUrl);
      router.refresh(); // Flush server component cache so Header shows user
    } catch {
      setError(ERROR_MESSAGES.Default);
    } finally {
      setLoading(false);
    }
  }

  async function handleGoogleSignIn() {
    setError(null);
    setGoogleLoading(true);
    try {
      // For OAuth providers like Google, we MUST allow redirect. 
      // NextAuth will handle the HTTP navigation to accounts.google.com.
      await signIn("google", { callbackUrl });
    } catch {
      setError(t("err_google_signin"));
      setGoogleLoading(false);
    }
  }

  // ── render ─────────────────────────────────────────────────────────────────

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
        <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>{t("welcome_back_title")}</h1>
        <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>{t("sign_in_subtitle")}</p>
      </div>

      {/* Card */}
      <div className="rounded-2xl p-6" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
        <form onSubmit={handleSubmit} className="space-y-4">

          {/* Error banner */}
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="flex flex-col gap-2 text-sm px-3 py-2.5 rounded-lg"
              style={{ background: "var(--surface-1)", border: "1px solid var(--surface-1)", color: "#fca5a5" }}
            >
              <div className="flex items-start gap-2">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>

              {/* Resend verification — only shown for pending-account errors */}
              {isPendingError && (
                resendSent
                  ? (
                    <div className="flex items-center gap-1.5 text-xs ml-6"
                         style={{ color: "#6ee7b7" }}>
                      <CheckCircle size={13} />
                      <span>{t("verification_sent")}</span>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={handleResend}
                      disabled={resendLoading}
                      className="ml-6 self-start text-xs underline underline-offset-2 hover:opacity-80 disabled:opacity-50 flex items-center gap-1"
                      style={{ color: "#fcd34d" }}
                    >
                      {resendLoading && <Loader2 size={11} className="animate-spin" />}
                      {t("resend_verification")}
                    </button>
                  )
              )}
            </motion.div>
          )}

          {/* Email field */}
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
                style={{
                  background:  "var(--surface-2)",
                  border:      "1px solid var(--border)",
                  color:       "var(--text-primary)",
                }}
                onFocus={e => (e.target.style.borderColor = "var(--brand-gold)")}
                onBlur={e  => (e.target.style.borderColor = "var(--border)")}
              />
            </div>
          </div>

          {/* Password field */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium" style={{ color: "var(--text-secondary)" }}>{t("password_label")}</label>
              <Link
                href="/forgot-password"
                className="text-xs hover:underline"
                style={{ color: "var(--text-muted)" }}
              >
                {t("forgot_password")}
              </Link>
            </div>
            <div className="relative">
              <Lock size={16} className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
                   style={{ color: "var(--text-muted)" }} />
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                placeholder={t("password_placeholder")}
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

          {/* Submit button */}
          <button
            type="submit"
            disabled={loading || googleLoading}
            className="w-full py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 transition-opacity disabled:opacity-60"
            style={{ background: "var(--brand-gold)", color: "#0d1520" }}
          >
            {loading
              ? <><Loader2 size={16} className="animate-spin" /> {t("signing_in")}</>
              : t("sign_in_password")}
          </button>

          {googleEnabled && (
            <>
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
            </>
          )}

        </form>
      </div>

      {/* Footer links */}
      <p className="text-center text-sm mt-4" style={{ color: "var(--text-muted)" }}>
        {t("no_account")} {" "}
        <Link href="/register" className="font-medium hover:underline" style={{ color: "var(--brand-gold)" }}>
          {t("create_one")}
        </Link>
      </p>
    </motion.div>
  );
}
