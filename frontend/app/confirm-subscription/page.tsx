/**
 * app/confirm-subscription/page.tsx
 * =====================================
 * Subscription confirmation landing page.
 *
 * FLOW:
 *   1. User submits email on the homepage ? POST /api/subscribers
 *      Backend generates a UUID confirm_token and (if RESEND_API_KEY is set)
 *      fires a transactional email with a link to this page:
 *        https://<APP_BASE_URL>/confirm-subscription?token=<uuid>
 *   2. User clicks the link ? lands here
 *   3. This page calls GET /api/subscribers/confirm?token=<uuid>  (server-side)
 *      Backend sets status="confirmed" and returns HTTPStatus.
 *   4. Show a branded success or error card based on the HTTP status.
 *
 * WHY Server Component (not client-side fetch):
 *   The confirmation is a one-shot action  no interactivity needed after load.
 *   Doing it server-side avoids a client-side loading spinner, works without JS,
 *   and makes the page fully SSR'd (better SEO and security).
 *
 * WHY fetch to host directly (not through Next.js proxy):
 *   Server Components run on the Node.js process  CORS doesn't apply.
 *   Calling 127.0.0.1:8000 directly is faster than going through the proxy.
 *   NEXT_PUBLIC vars are not available server-side without the NEXT_PUBLIC prefix;
 *   use process.env.DEEPCOIN_INTERNAL_API_URL (defaults to http://127.0.0.1:8000).
 */

import Link         from "next/link";
import { CheckCircle, XCircle, AlertCircle, ArrowLeft } from "lucide-react";

// -- Types --------------------------------------------------------------------

type ConfirmStatus = "success" | "not_found" | "no_token" | "error";

// -- Server-side confirmation call --------------------------------------------

async function confirmToken(token: string): Promise<ConfirmStatus> {
  const base = process.env.DEEPCOIN_INTERNAL_API_URL ?? "http://127.0.0.1:8000";
  try {
    const res = await fetch(
      `${base}/api/subscribers/confirm?token=${encodeURIComponent(token)}`,
      // next: { revalidate: 0 } ensures no stale cache for a one-time action
      { cache: "no-store" },
    );
    // Backend returns 200 for both "new confirmation" AND "already confirmed"
    //  both are positive outcomes, show the same success card.
    if (res.status === 200) return "success";
    // 400 = invalid token (not found / already used)
    if (res.status === 400) return "not_found";
    return "error";
  } catch {
    return "error";
  }
}

// -- UI sub-components --------------------------------------------------------

function Card({
  icon,
  title,
  body,
  cta,
}: {
  icon:  React.ReactNode;
  title: string;
  body:  string;
  cta:   React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center px-4"
         style={{ backgroundColor: "var(--bg)" }}>
      <div
        className="w-full max-w-md rounded-2xl border p-8 text-center space-y-4"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
      >
        <div className="flex justify-center">{icon}</div>
        <h1 className="text-xl font-bold" style={{ color: "var(--text-primary)" }}>{title}</h1>
        <p className="text-sm leading-relaxed" style={{ color: "var(--text-muted)" }}>{body}</p>
        <div className="pt-2">{cta}</div>
        <Link
          href="/"
          className="flex items-center justify-center gap-1.5 text-xs mt-4 hover:underline"
          style={{ color: "var(--text-muted)" }}
        >
          <ArrowLeft size={12} /> Back to DeepCoin
        </Link>
      </div>
    </div>
  );
}

// -- Page ---------------------------------------------------------------------

export default async function ConfirmSubscriptionPage({
  searchParams,
}: {
  searchParams: Promise<{ token?: string }>;
}) {
  const { token } = await searchParams;

  if (!token) {
    return (
      <Card
        icon={<AlertCircle size={40} style={{ color: "#f59e0b" }} />}
        title="Invalid link"
        body="This confirmation link is missing a token. Please use the link sent to your inbox."
        cta={null}
      />
    );
  }

  const status = await confirmToken(token);

  if (status === "success") {
    return (
      <Card
        icon={<CheckCircle size={40} style={{ color: "#22c55e" }} />}
        title="Subscription confirmed!"
        body="You're on the DeepCoin mailing list. We'll notify you about the public API launch, new coin types, and model updates."
        cta={
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold"
            style={{ backgroundColor: "var(--brand-gold)", color: "var(--surface-0)" }}
          >
            Explore DeepCoin
          </Link>
        }
      />
    );
  }

  if (status === "not_found") {
    return (
      <Card
        icon={<XCircle size={40} style={{ color: "#f87171" }} />}
        title="Link not found"
        body="This confirmation link is invalid or has expired. Try signing up again from the homepage."
        cta={
          <Link
            href="/#waitlist"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold"
            style={{ backgroundColor: "var(--surface-2)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            Sign up again
          </Link>
        }
      />
    );
  }

  // Generic error fallback
  return (
    <Card
      icon={<XCircle size={40} style={{ color: "#f87171" }} />}
      title="Something went wrong"
      body="Unable to confirm your subscription right now. Please try again in a few minutes or contact support."
      cta={null}
    />
  );
}

