/**
 * app/confirm-subscription/page.tsx
 * ===================================
 * Handles the confirmation link that appears in the subscription email
 * (or the inline dev-mode button in EmailCapture.tsx when no SMTP is set).
 *
 * WHAT:
 *   Reads ?token=xxx from the URL and POSTs to
 *   GET /api/subscribers/confirm?token=xxx on the FastAPI backend.
 *   Renders a branded success or error state.
 *
 * WHY a Next.js page instead of a direct FastAPI link:
 *   The FastAPI endpoint already returns a standalone HTML page for email
 *   clients that open the link directly (no JS). The Next.js page wraps
 *   the same action in the app shell, giving a better experience when
 *   the user is already in the browser (nav, dark mode, etc.).
 *
 * NOTE: This page is public (no auth required). The token itself is the
 *       credential — UUID4 has 122 bits of entropy.
 */

"use client";

import { useEffect, useState } from "react";
import { useSearchParams }     from "next/navigation";
import { Suspense }            from "react";
import Link                    from "next/link";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";

/* ─── inner client component (needs useSearchParams) ─────────────────────── */

function ConfirmPage() {
  const params = useSearchParams();
  const token  = params.get("token") ?? "";

  type Status = "loading" | "confirmed" | "error";
  const [status,  setStatus]  = useState<Status>("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!token) {
      setStatus("error");
      setMessage("No confirmation token found in this URL. Please use the link from your email.");
      return;
    }

    const CLASSIFY_URL = process.env.NEXT_PUBLIC_CLASSIFY_URL ?? "http://127.0.0.1:8000";

    fetch(`${CLASSIFY_URL}/api/subscribers/confirm?token=${encodeURIComponent(token)}`)
      .then(async (res) => {
        if (res.ok) {
          setStatus("confirmed");
        } else {
          // FastAPI returns HTML for this endpoint — extract text from body
          const text = await res.text().catch(() => "");
          const match = text.match(/<p[^>]*>(.*?)<\/p>/);
          setMessage(match?.[1]?.replace(/<[^>]+>/g, "") ?? "This confirmation link is invalid or has already been used.");
          setStatus("error");
        }
      })
      .catch(() => {
        setMessage("Could not reach the server. Please try again.");
        setStatus("error");
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  return (
    <main
      className="min-h-screen flex flex-col items-center justify-center px-6 py-20 text-center"
      style={{ backgroundColor: "var(--bg-base, #070d1a)", color: "var(--text-primary, #f1f5f9)" }}
    >
      {status === "loading" && (
        <>
          <Loader2 size={48} className="animate-spin mb-6" style={{ color: "#d4a853" }} />
          <p style={{ color: "var(--text-secondary, #94a3b8)" }}>Confirming your subscription…</p>
        </>
      )}

      {status === "confirmed" && (
        <>
          <CheckCircle size={56} className="mb-6" style={{ color: "#10b981" }} />
          <h1 className="text-3xl font-black mb-3" style={{ color: "#d4a853" }}>
            Subscription confirmed!
          </h1>
          <p className="max-w-sm text-sm mb-8" style={{ color: "var(--text-secondary, #94a3b8)" }}>
            You&rsquo;re on the list. We&rsquo;ll send one email per major release — no spam,
            unsubscribe any time via the link in any email.
          </p>
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-sm transition-all hover:brightness-110"
            style={{ backgroundColor: "#d4a853", color: "#0a1628" }}
          >
            Back to DeepCoin
          </Link>
        </>
      )}

      {status === "error" && (
        <>
          <XCircle size={56} className="mb-6" style={{ color: "#ef4444" }} />
          <h1 className="text-3xl font-black mb-3">Something went wrong</h1>
          <p className="max-w-sm text-sm mb-8" style={{ color: "var(--text-secondary, #94a3b8)" }}>
            {message || "This confirmation link is invalid or has already been used."}
          </p>
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-sm transition-all hover:brightness-110"
            style={{ backgroundColor: "#d4a853", color: "#0a1628" }}
          >
            Back to DeepCoin
          </Link>
        </>
      )}

      {/* DeepCoin wordmark */}
      <p className="mt-16 text-xs font-bold tracking-widest uppercase" style={{ color: "rgba(212,168,83,0.4)" }}>
        DeepCoin · AI Numismatic Analysis
      </p>
    </main>
  );
}

/* ─── page shell — wraps useSearchParams in Suspense (Next.js 15 requirement) */

export default function ConfirmSubscriptionPage() {
  return (
    <Suspense
      fallback={
        <main className="min-h-screen flex items-center justify-center">
          <Loader2 size={40} className="animate-spin" style={{ color: "#d4a853" }} />
        </main>
      }
    >
      <ConfirmPage />
    </Suspense>
  );
}
