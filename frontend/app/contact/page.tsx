"use client";

/**
 * app/contact/page.tsx
 * =====================
 * Contact form for DeepCoin.
 *
 * WHAT: A styled form that lets any visitor send a message to the project team.
 *       On submit it opens the user's email client with the message pre-filled
 *       via a `mailto:` link — zero backend required, works in every browser.
 *
 * WHY mailto (not a POST endpoint):
 *   - No SMTP/Resend config needed in development
 *   - Message arrives in the author's real inbox from the sender's own account
 *     (so replies go directly to the sender — no reply-to juggling)
 *   - Transparent: the user sees exactly what will be sent before hitting Send
 *
 * WHO this is for: researchers, museum partners, recruiters, thesis evaluators.
 *
 * FIELDS:
 *   name    — sender's name (used in the email body)
 *   email   — pre-filled in mailto: "to" but the client can override
 *   subject — mapped to mailto: "subject"
 *   message — mapped to mailto: "body"
 */

import { useState }       from "react";
import Link               from "next/link";
import { Mail, Send, ExternalLink, User, MessageSquare, FileText } from "lucide-react";

const ADMIN_EMAIL = "dhia.chaieb@esprit.tn";

// Pre-built subject options to help users pick a clear topic
const SUBJECTS = [
  "Research collaboration",
  "Access to the public API",
  "Bug report / feedback",
  "Dataset / licensing inquiry",
  "Press / media",
  "Other",
];

export default function ContactPage() {
  const [name,    setName]    = useState("");
  const [email,   setEmail]   = useState("");
  const [subject, setSubject] = useState(SUBJECTS[0]);
  const [message, setMessage] = useState("");
  const [sent,    setSent]    = useState(false);

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();

    const body = [
      `From: ${name} <${email}>`,
      "",
      message,
      "",
      "— Sent via DeepCoin contact form",
    ].join("\n");

    const mailto =
      `mailto:${ADMIN_EMAIL}` +
      `?subject=${encodeURIComponent(`[DeepCoin] ${subject}`)}` +
      `&body=${encodeURIComponent(body)}`;

    window.location.href = mailto;
    setSent(true);
  }

  const inputCls =
    "w-full rounded-lg px-3.5 py-2.5 text-sm outline-none transition-colors " +
    "focus:ring-2 focus:ring-[#d4a853]/50";
  const inputStyle = {
    backgroundColor: "var(--surface-2)",
    border:          "1px solid var(--border)",
    color:           "var(--text-primary)",
  };

  return (
    <main className="min-h-screen py-16 px-4" style={{ backgroundColor: "var(--bg)" }}>
      <div className="max-w-xl mx-auto space-y-8">

        {/* Header */}
        <div className="text-center space-y-2">
          <div
            className="inline-flex items-center justify-center w-12 h-12 rounded-xl mb-2"
            style={{ backgroundColor: "var(--surface-1)", border: "1px solid var(--border)" }}
          >
            <Mail size={22} style={{ color: "var(--brand-gold)" }} />
          </div>
          <h1
            className="text-2xl font-black"
            style={{ color: "var(--text-primary)" }}
          >
            Contact the team
          </h1>
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            Questions about the API, research proposals, dataset access, bug reports&nbsp;—
            we&rsquo;d love to hear from you.
          </p>
        </div>

        {/* Form card */}
        <div
          className="rounded-2xl border p-6 space-y-5"
          style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
        >
          {sent ? (
            /* ── Post-submit state ── */
            <div className="py-6 text-center space-y-3">
              <Send size={32} className="mx-auto" style={{ color: "#22c55e" }} />
              <p className="font-semibold" style={{ color: "var(--text-primary)" }}>
                Your email client should have opened.
              </p>
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                If nothing happened, email us directly at{" "}
                <a
                  href={`mailto:${ADMIN_EMAIL}`}
                  className="underline"
                  style={{ color: "var(--brand-gold)" }}
                >
                  {ADMIN_EMAIL}
                </a>
              </p>
              <button
                onClick={() => setSent(false)}
                className="text-xs underline mt-2"
                style={{ color: "var(--text-muted)" }}
              >
                Send another message
              </button>
            </div>
          ) : (
            /* ── Form ── */
            <form onSubmit={handleSubmit} className="space-y-4">

              {/* Name */}
              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-xs font-semibold"
                       style={{ color: "var(--text-secondary)" }}>
                  <User size={11} /> Your name
                </label>
                <input
                  type="text"
                  required
                  placeholder="Dhia Chaieb"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  className={inputCls}
                  style={inputStyle}
                />
              </div>

              {/* Email */}
              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-xs font-semibold"
                       style={{ color: "var(--text-secondary)" }}>
                  <Mail size={11} /> Your email
                </label>
                <input
                  type="email"
                  required
                  placeholder="you@example.com"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className={inputCls}
                  style={inputStyle}
                />
              </div>

              {/* Subject */}
              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-xs font-semibold"
                       style={{ color: "var(--text-secondary)" }}>
                  <FileText size={11} /> Subject
                </label>
                <select
                  value={subject}
                  onChange={e => setSubject(e.target.value)}
                  className={inputCls}
                  style={inputStyle}
                >
                  {SUBJECTS.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>

              {/* Message */}
              <div className="space-y-1.5">
                <label className="flex items-center gap-1.5 text-xs font-semibold"
                       style={{ color: "var(--text-secondary)" }}>
                  <MessageSquare size={11} /> Message
                </label>
                <textarea
                  required
                  rows={5}
                  placeholder="Tell us what's on your mind…"
                  value={message}
                  onChange={e => setMessage(e.target.value)}
                  className={inputCls}
                  style={{ ...inputStyle, resize: "vertical" }}
                />
              </div>

              {/* Submit */}
              <button
                type="submit"
                disabled={!name || !email || !message}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg
                           text-sm font-bold transition-opacity disabled:opacity-40"
                style={{ backgroundColor: "var(--brand-gold)", color: "#0a1628" }}
              >
                <Send size={14} />
                Open in email client
              </button>

              <p className="text-[11px] text-center" style={{ color: "var(--text-muted)" }}>
                This opens your default email application pre-filled with your message.
              </p>
            </form>
          )}
        </div>

        {/* Direct email fallback */}
        <div className="text-center">
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            Prefer to email directly?{" "}
            <a
              href={`mailto:${ADMIN_EMAIL}`}
              className="inline-flex items-center gap-1 underline"
              style={{ color: "var(--text-secondary)" }}
            >
              {ADMIN_EMAIL} <ExternalLink size={10} />
            </a>
          </p>
        </div>

        {/* Back link */}
        <div className="text-center">
          <Link href="/" className="text-xs hover:underline" style={{ color: "var(--text-muted)" }}>
            ← Back to DeepCoin
          </Link>
        </div>

      </div>
    </main>
  );
}
