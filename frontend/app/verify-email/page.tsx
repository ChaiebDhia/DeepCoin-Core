/**
 * app/verify-email/page.tsx
 * ==========================
 * /verify-email?token=<one-time-token>
 *
 * This URL is embedded in the account verification email.
 * When the user clicks the link, this Server Component renders and
 * immediately calls the FastAPI verify-email endpoint SERVER-SIDE.
 *
 * WHY a Server Component (not a client fetch):
 *   1. The verification is a one-time, idempotent action.  Doing it at
 *      render time (not in a useEffect) means:
 *      - The result is available for the initial HTML render (no loading flicker)
 *      - No second round-trip needed after hydration
 *      - Works even if the user has JavaScript disabled
 *
 *   2. The FastAPI URL is not exposed to the browser (no NEXT_PUBLIC_ prefix
 *      needed) — the server-side fetch stays within the internal network.
 *
 * FLOW:
 *   1. User clicks the link in their verification email.
 *   2. Next.js renders this component on the server.
 *   3. Server calls GET /auth/verify-email?token=<tok>.
 *   4. FastAPI activates the account (status=active) and marks the token used.
 *   5. User sees either a branded success page or a clear error with next steps.
 *
 * HOW token validation works (FastAPI side):
 *   - Token must exist in the email_verifications table
 *   - Token must not be expired (48 h window)
 *   - Token must not be marked as used (one-time use)
 *   On success the user's status is set to "active" and they can sign in.
 */

import type { Metadata } from "next";
import Link              from "next/link";
import { CheckCircle2, XCircle, Coins } from "lucide-react";

export const metadata: Metadata = {
  title:       "Email Verified · DeepCoin",
  description: "Confirming your DeepCoin account email address.",
};

const FASTAPI_URL = process.env.AUTH_FASTAPI_URL ?? "http://127.0.0.1:8000";

interface PageProps {
  searchParams: Promise<{ token?: string }>;
}

export default async function VerifyEmailPage({ searchParams }: PageProps) {
  const { token } = await searchParams;

  // ── Guard: token absent ───────────────────────────────────────────────────
  if (!token) {
    return (
      <VerifyLayout>
        <StatusCard
          ok={false}
          title="Missing verification token"
          body="The verification link is incomplete. Please click the full link from your email, or request a new one."
          action={{ href: "/login", label: "Back to sign in" }}
        />
      </VerifyLayout>
    );
  }

  // ── Call FastAPI server-side ──────────────────────────────────────────────
  let ok      = false;
  let message = "Verification failed.";

  try {
    const res = await fetch(
      `${FASTAPI_URL}/auth/verify-email?token=${encodeURIComponent(token)}`,
      {
        // next.js does not cache this response — each token is single-use,
        // so we must always hit the FastAPI endpoint fresh.
        cache: "no-store",
      }
    );
    const body = (await res.json()) as { message?: string; detail?: string };

    if (res.ok) {
      ok      = true;
      message = body.message ?? "Email verified successfully!";
    } else {
      message = body.detail ?? `Verification failed (HTTP ${res.status}).`;
    }
  } catch {
    message = "Could not reach the verification server. Please try again later.";
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <VerifyLayout>
      {ok ? (
        <StatusCard
          ok={true}
          title="Email verified!"
          body={message}
          action={{ href: "/login", label: "Sign in to your account" }}
          sub="Your account is now active. Welcome to DeepCoin."
        />
      ) : (
        <StatusCard
          ok={false}
          title="Verification failed"
          body={message}
          action={{ href: "/forgot-password", label: "Request a new link" }}
          sub="Links expire after 48 hours and can only be used once."
        />
      )}
    </VerifyLayout>
  );
}

// ── Layout wrapper ────────────────────────────────────────────────────────────

function VerifyLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-8rem)] py-12">
      <div className="w-full max-w-sm mx-auto">
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
        </div>
        {children}
      </div>
    </div>
  );
}

// ── StatusCard ────────────────────────────────────────────────────────────────

function StatusCard({
  ok, title, body, sub, action,
}: {
  ok:     boolean;
  title:  string;
  body:   string;
  sub?:   string;
  action: { href: string; label: string };
}) {
  return (
    <div
      className="rounded-2xl p-8 text-center space-y-4"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
    >
      <div
        className="inline-flex items-center justify-center w-14 h-14 rounded-full mx-auto"
        style={{ background: ok ? "rgba(16,185,129,0.12)" : "rgba(239,68,68,0.10)" }}
      >
        {ok
          ? <CheckCircle2 size={30} style={{ color: "#10b981" }} />
          : <XCircle     size={30} style={{ color: "#f87171" }} />}
      </div>

      <div className="space-y-1.5">
        <h1 className="text-xl font-bold" style={{ color: "var(--text-primary)" }}>
          {title}
        </h1>
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>{body}</p>
        {sub && (
          <p className="text-xs mt-1" style={{ color: "var(--text-muted)", opacity: 0.7 }}>{sub}</p>
        )}
      </div>

      <Link
        href={action.href}
        className="inline-block w-full py-2.5 rounded-lg text-sm font-semibold transition-opacity hover:opacity-90"
        style={{ background: "var(--brand-gold)", color: "#0d1520" }}
      >
        {action.label}
      </Link>
    </div>
  );
}
