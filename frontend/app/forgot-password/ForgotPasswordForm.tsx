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
import { useTranslations } from "next-intl";
import { forgotPassword } from "@/lib/api";

function toErrorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item && typeof (item as { msg?: unknown }).msg === "string") {
          return (item as { msg: string }).msg;
        }
        return "";
      })
      .filter(Boolean);

    if (messages.length) return messages.join(" · ");
  }

  if (detail && typeof detail === "object") {
    const candidate = detail as { detail?: unknown; msg?: unknown; message?: unknown };
    if (typeof candidate.detail === "string" && candidate.detail.trim()) return candidate.detail;
    if (candidate.detail && typeof candidate.detail === "object") {
      const nested = candidate.detail as { msg?: unknown };
      if (typeof nested.msg === "string" && nested.msg.trim()) return nested.msg;
    }
    if (typeof candidate.msg === "string" && candidate.msg.trim()) return candidate.msg;
    if (typeof candidate.message === "string" && candidate.message.trim()) return candidate.message;
  }

  return fallback;
}

export default function ForgotPasswordForm() {
  const t = useTranslations("ForgotPassword");
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
    } catch (err: unknown) {
      setError(toErrorMessage(err, t("err_submit")));
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
          {t("title")}
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
          {t("subtitle")}
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
                  {t("success_title")}
                </p>
                <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                  {t("success_message_prefix")} <strong>{email}</strong>{t("success_message_mid")}
                  {t("success_message_suffix")}
                </p>              <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>
                <span className="text-[10px] block">{t("tips_title")}</span>
                {t("tip_spam")}  <br/>
                {t("tip_verify")}  <br/>
                {t("tip_support")}
              </p>              </div>
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                {t("didnt_receive")} {" "}
                <button
                  onClick={() => { setSent(false); setEmail(""); }}
                  className="underline hover:opacity-80"
                  style={{ color: "var(--brand-gold)" }}
                >
                  {t("try_different")}
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
                  {t("email_address")}
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
                    placeholder={t("email_placeholder")}
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
                  ? <><Loader2 size={16} className="animate-spin" /> {t("sending")}</>
                  : t("send_link")}
              </button>
            </motion.form>
          )}
        </AnimatePresence>
      </div>

      {/* Footer links */}
      <p className="text-center text-sm mt-4" style={{ color: "var(--text-muted)" }}>
        {t("remembered_it")} {" "}
        <Link href="/login" className="font-medium hover:underline" style={{ color: "var(--brand-gold)" }}>
          {t("sign_in")}
        </Link>
      </p>
    </motion.div>
  );
}
