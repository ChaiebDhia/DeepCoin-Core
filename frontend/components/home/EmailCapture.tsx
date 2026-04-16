"use client";

/**
 * components/home/EmailCapture.tsx
 * ==================================
 * Waitlist / mailing-list email capture section.
 *
 * WHAT: Animated form with email input + submit button.
 *       Three states: idle → loading (spinner) → done (success message).
 *
 * SUCCESS STATES:
 *   The backend returns { ok, message, email_sent } on a successful subscribe.
 *
 *   • email_sent=true  (RESEND_API_KEY / SMTP set in production):
 *       "We'll reach out to <email> when we launch"
 *
 *   • email_sent=false (dev / no SMTP — current PFE environment):
 *       "{t("we_saved")} <email>. You'll be the first to know when we launch."
 *
 * WHY no confirm link:
 *   The previous design showed a "Confirm subscription →" link pointing to
 *   /confirm-subscription?token=xxx when SMTP was not configured.  This was
 *   dead UX: in production there's no SMTP, so no email arrives, and the
 *   page was only reachable via the manually constructed URL.  A waitlist
 *   does not require double opt-in (that's a legal requirement for marketing
 *   newsletters, not launch-notification lists).  The simpler "You're on the
 *   list!" message is honest and requires no extra click.
 *
 * BACKED BY:
 *   POST   /api/subscribers                     — subscribe (FastAPI)
 *   GET    /api/subscribers/confirm?token=xxx   — confirm (admin utility, SMTP future)
 *   GET    /api/subscribers/unsubscribe?token=xxx — unsubscribe (SMTP future)
 */

import { useState }              from "react";
import { useSession }            from "next-auth/react";
import { motion, AnimatePresence } from "framer-motion";
import { Mail, CheckCircle, Loader2, ArrowRight } from "lucide-react";
import { useTranslations } from "next-intl";

type State = "idle" | "loading" | "done";

interface SubscribeResponse {
  ok:         boolean;
  message:    string;
  email_sent: boolean;
}

export function EmailCapture() {
  const t = useTranslations("EmailCapture");
  const { data: session } = useSession();
  const [email,     setEmail]     = useState("");
  const [state,     setState]     = useState<State>("idle");
  const [error,     setError]     = useState("");
  const [emailSent, setEmailSent] = useState(false);

  const isAuthed = !!session?.user?.email;
  const submitEmail = isAuthed ? session.user.email : email;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!submitEmail || !submitEmail.includes("@")) {
      setError(t("err_invalid"));
      return;
    }
    setError("");
    setState("loading");

    try {
      const res = await fetch("/api/subscribers", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ email: submitEmail }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({})) as { detail?: string };
        setError(data.detail ?? t("err_submit"));
        setState("idle");
        return;
      }

      const data: SubscribeResponse = await res.json();
      setEmailSent(data.email_sent ?? false);
    } catch {
      setError(t("err_server"));
      setState("idle");
      return;
    }

    setState("done");
  }

  return (
    <section className="py-24">
      <div
        className="relative rounded-3xl overflow-hidden p-px"
        style={{
          background:
            "linear-gradient(135deg, rgba(212,168,83,0.5) 0%, rgba(59,130,246,0.3) 50%, rgba(212,168,83,0.5) 100%)",
        }}
      >
        {/* Inner surface */}
        <div
          className="relative rounded-3xl px-8 py-16 text-center overflow-hidden bg-white shadow-xl dark:bg-[var(--surface-1)] dark:shadow-none"
        >
          {/* Radial gold glow background accent */}
          <div
            className="absolute inset-0 pointer-events-none"
            aria-hidden
            style={{
              background:
                "radial-gradient(ellipse 70% 60% at 50% 50%, var(--brand-gold-10) 0%, transparent 70%)",
            }}
          />

          {/* Mail icon */}
          <div className="relative z-10 flex justify-center mb-6">
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center animate-glow-pulse"
              style={{ background: "rgba(212,168,83,0.15)", color: "var(--brand-gold)" }}
            >
              <Mail size={26} />
            </div>
          </div>

          {/* Heading */}
          <h2
            className="relative z-10 text-3xl sm:text-4xl font-black mb-3"
            style={{ color: "var(--text-primary)" }}
          >
            {t("hero_title")}
          </h2>
          <p
            className="relative z-10 max-w-md mx-auto text-sm mb-10"
            style={{ color: "var(--text-secondary)" }}
          >
            Get notified when the public API launches, new coin types are added
            to the knowledge base, or a new model version is released.
          </p>

          {/* Form / Success swap */}
          <AnimatePresence mode="wait">
            {state !== "done" ? (
              <motion.form
                key="form"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.3 }}
                onSubmit={handleSubmit}
                className="relative z-10 flex flex-col sm:flex-row items-stretch gap-3 max-w-md mx-auto"
              >
                <div className="flex flex-col items-center w-full max-w-sm mb-4">
                  {isAuthed && (
                    <p className="text-sm mb-3" style={{ color: "var(--text-secondary)" }}>
                      {t("subscribe_as")} <strong>{session.user.email}</strong>
                    </p>
                  )}
                  {error && (
                    <p className="text-red-500 text-xs mb-3 font-semibold">{error}</p>
                  )}
                </div>
                {!isAuthed && (
                  <input
                    type="email"
                    placeholder={t("placeholder")}
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={state === "loading"}
                    className="flex-1 px-4 py-3 rounded-xl border text-sm outline-none focus:ring-2 focus:ring-[var(--brand-gold)] disabled:opacity-50 transition-all"
                    style={{
                      backgroundColor: "var(--surface-2)",
                      borderColor:     error ? "#ef4444" : "var(--border)",
                      color:           "var(--text-primary)",
                    }}
                  />
                )}
                <button
                  type="submit"
                  disabled={state === "loading"}
                  className={`inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold text-sm transition-all duration-200 hover:brightness-110 disabled:opacity-60 flex-shrink-0 ${isAuthed ? 'w-full max-w-xs' : ''}`}
                  style={{ backgroundColor: "var(--brand-gold)", color: "var(--brand-navy)" }}
                >
                  {state === "loading" ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : (
                    <>
                      {t("btn_notify")}
                      <ArrowRight size={14} />
                    </>
                  )}
                </button>
              </motion.form>
            ) : (
              <motion.div
                key="done"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.35, type: "spring", stiffness: 200 }}
                className="relative z-10 flex flex-col items-center gap-3"
              >
                {/* Single success state — email confirmed or queued */}
                <>
                  <CheckCircle size={40} style={{ color: "#10b981" }} />
                  <p className="font-bold text-base" style={{ color: "var(--text-primary)" }}>
                    {t("success_title")}
                  </p>
                  <p className="text-sm max-w-xs" style={{ color: "var(--text-secondary)" }}>
                    {emailSent
                      ? <>{t("reach_out")} <strong>{submitEmail}</strong> {t("reach_out_end")}</>
                      : <>{t("we_saved")} <strong>{submitEmail}</strong>{t("we_saved_end")}</>}
                  </p>
                  <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                    {t("privacy")}
                  </p>
                </>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Inline error */}
          {error && state === "idle" && (
            <p className="relative z-10 mt-2 text-xs" style={{ color: "#ef4444" }}>
              {error}
            </p>
          )}

          {/* Privacy note */}
          {state !== "done" && (
            <p className="relative z-10 mt-4 text-xs" style={{ color: "var(--text-muted)" }}>
              {t("privacy")}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}


