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
import { signIn }              from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";
import Link                    from "next/link";
import { motion }              from "framer-motion";
import { Mail, Lock, Coins, AlertCircle, Loader2 } from "lucide-react";

export function LoginForm() {
  const router       = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl  = searchParams.get("callbackUrl") ?? "/";

  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [error,    setError]    = useState<string | null>(null);
  const [loading,  setLoading]  = useState(false);

  // ── friendly error messages ────────────────────────────────────────────────

  const ERROR_MESSAGES: Record<string, string> = {
    CredentialsSignin:   "Incorrect email or password. Please try again.",
    CallbackRouteError:  "Please verify your email address before signing in. Check your inbox for a verification link.",
    Default:             "Something went wrong. Please try again.",
  };

  // ── submit handler ─────────────────────────────────────────────────────────

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
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
             style={{ background: "linear-gradient(135deg, rgba(212,175,55,0.15) 0%, rgba(212,175,55,0.05) 100%)", border: "1px solid rgba(212,175,55,0.3)" }}>
          <Coins size={28} style={{ color: "var(--brand-gold)" }} />
        </div>
        <h1 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
          Welcome back
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--text-muted)" }}>
          Sign in to your DeepCoin account
        </p>
      </div>

      {/* Card */}
      <div className="rounded-2xl p-6" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
        <form onSubmit={handleSubmit} className="space-y-4">

          {/* Error banner */}
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="flex items-start gap-2 text-sm px-3 py-2.5 rounded-lg"
              style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#fca5a5" }}
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
                autoComplete="current-password"
                placeholder="••••••••"
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
            disabled={loading}
            className="w-full py-2.5 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 transition-opacity disabled:opacity-60"
            style={{ background: "var(--brand-gold)", color: "#0d1520" }}
          >
            {loading
              ? <><Loader2 size={16} className="animate-spin" /> Signing in…</>
              : "Sign in"}
          </button>

        </form>
      </div>

      {/* Footer links */}
      <p className="text-center text-sm mt-4" style={{ color: "var(--text-muted)" }}>
        No account?{" "}
        <Link href="/register" className="font-medium hover:underline" style={{ color: "var(--brand-gold)" }}>
          Create one
        </Link>
      </p>
    </motion.div>
  );
}
