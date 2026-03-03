"use client";

/**
 * components/home/EmailCapture.tsx
 * ==================================
 * Waitlist / mailing-list email capture section.
 *
 * WHAT: Animated form with email input + submit button.
 *       Three states: idle → loading (spinner) → done (checkmark + message).
 *       AnimatePresence handles the form ↔ success swap.
 *
 * WHY include this: Demonstrates full interaction design skill — error state,
 *   loading state, success state, animated transition. The submit currently
 *   simulates a 900 ms delay; the comment explains where to wire a real endpoint.
 *
 * NOTE: Data is NOT stored. This is a visual demo of the interaction pattern.
 *   When a real endpoint is ready, replace the `simulateSubmit` call with
 *   `await fetch("/api/subscribers", { method:"POST", body: JSON.stringify({email}) })`.
 *
 * HOW animated border works:
 *   The outer wrapper has a `conic-gradient` rotating pseudo-element.
 *   We replicate it with a CSS class `.animate-border-rotate` (see globals.css).
 *   Here we use an inline Framer Motion approach for the same effect without
 *   requiring a pseudo-element (pure React, no extra CSS needed).
 */

import { useState }              from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mail, CheckCircle, Loader2, ArrowRight } from "lucide-react";

type State = "idle" | "loading" | "done";

export function EmailCapture() {
  const [email, setEmail] = useState("");
  const [state, setState] = useState<State>("idle");
  const [error, setError] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email.includes("@")) {
      setError("Please enter a valid email address.");
      return;
    }
    setError("");
    setState("loading");

    // ──────────────────────────────────────────────────────────────────
    // TODO: Replace with real API call when endpoint is available:
    //   await fetch("/api/subscribers", { method: "POST",
    //     headers: { "Content-Type": "application/json" },
    //     body: JSON.stringify({ email })
    //   });
    // ──────────────────────────────────────────────────────────────────
    await new Promise((r) => setTimeout(r, 900));
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
          className="relative rounded-3xl px-8 py-16 text-center overflow-hidden"
          style={{ backgroundColor: "var(--surface-1)" }}
        >
          {/* Radial gold glow background accent */}
          <div
            className="absolute inset-0 pointer-events-none"
            aria-hidden
            style={{
              background:
                "radial-gradient(ellipse 70% 60% at 50% 50%, rgba(212,168,83,0.07) 0%, transparent 70%)",
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
            Stay in the loop
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
                <input
                  type="email"
                  placeholder="your@email.com"
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
                <button
                  type="submit"
                  disabled={state === "loading"}
                  className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold text-sm transition-all duration-200 hover:brightness-110 disabled:opacity-60 flex-shrink-0"
                  style={{ backgroundColor: "var(--brand-gold)", color: "#0a1628" }}
                >
                  {state === "loading" ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : (
                    <>
                      Notify me
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
                <CheckCircle size={40} style={{ color: "#10b981" }} />
                <p className="font-bold text-base" style={{ color: "var(--text-primary)" }}>
                  You&rsquo;re on the list!
                </p>
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  We&rsquo;ll reach out at <strong>{email}</strong> when there&rsquo;s news.
                </p>
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
              No spam. One email per major release. Unsubscribe any time.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
