/**
 * components/ui/footer.tsx
 * ==========================
 * Site-wide footer — Server Component (no "use client" needed).
 *
 * WHAT: Four-column footer with product links, resource links, tech stack,
 *       and an about section. Bottom bar has copyright and GitHub link.
 *
 * WHY Server Component:
 *   No client interactivity — all links are static. Keeping this as a Server
 *   Component avoids shipping unnecessary JS for purely static HTML.
 *
 * HOW it fits:
 *   Imported in app/layout.tsx to replace the previous inline <footer>.
 *   All columns are rendered in a responsive grid (2 cols on mobile → 4 on desktop).
 */

import Link    from "next/link";
import { Github, ExternalLink } from "lucide-react";

const COLUMNS = [
  {
    heading: "Product",
    links: [
      { label: "Analyse a coin",   href: "/#analyse" },
      { label: "Analysis history", href: "/history" },
      { label: "How it works",     href: "/#how-it-works" },
      { label: "Features",         href: "/#features" },
    ],
  },
  {
    heading: "Resources",
    links: [
      { label: "Corpus Nummorum",  href: "https://www.corpus-nummorum.eu",  external: true },
      { label: "API Documentation",href: "/api/docs",                        external: true },
      { label: "Engineering Docs", href: "/admin" },
    ],
  },
];

const TECH = [
  "EfficientNet-B3 + PyTorch 2.6",
  "LangGraph + ChromaDB",
  "FastAPI + PostgreSQL",
  "Next.js 15 + Tailwind v4",
  "Framer Motion 12",
  "Docker Compose",
];

export function Footer() {
  return (
    <footer
      className="border-t mt-8"
      style={{ borderColor: "var(--border)" }}
    >
      <div className="mx-auto max-w-6xl px-5 pt-12 pb-8">
        {/* Columns */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-10 mb-12">
          {/* Product + Resources columns */}
          {COLUMNS.map(({ heading, links }) => (
            <div key={heading}>
              <h3 className="text-xs font-black uppercase tracking-widest mb-4" style={{ color: "var(--text-muted)" }}>
                {heading}
              </h3>
              <ul className="space-y-2.5">
                {links.map(({ label, href, external }) => (
                  <li key={label}>
                    {external ? (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs inline-flex items-center gap-1 hover:text-white transition-colors"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {label}
                        <ExternalLink size={10} className="opacity-60" />
                      </a>
                    ) : (
                      <Link
                        href={href}
                        className="text-xs hover:text-white transition-colors"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}

          {/* Tech stack */}
          <div>
            <h3 className="text-xs font-black uppercase tracking-widest mb-4" style={{ color: "var(--text-muted)" }}>
              Tech Stack
            </h3>
            <ul className="space-y-2.5">
              {TECH.map((t) => (
                <li key={t} className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  {t}
                </li>
              ))}
            </ul>
          </div>

          {/* About */}
          <div>
            <h3 className="text-xs font-black uppercase tracking-widest mb-4" style={{ color: "var(--text-muted)" }}>
              About
            </h3>
            <p className="text-xs leading-relaxed mb-4" style={{ color: "var(--text-secondary)" }}>
              PFE 2026 — Final Year Engineering Internship.
            </p>
            <p className="text-xs leading-relaxed mb-4" style={{ color: "var(--text-secondary)" }}>
              ESPRIT School of Engineering × YEBNI, Tunisia.
            </p>
            <p className="text-xs font-semibold" style={{ color: "var(--brand-gold)" }}>
              Dhia Chaieb
            </p>
            <a
              href="https://github.com/ChaiebDhia/DeepCoin-Core"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 mt-2 text-xs hover:text-white transition-colors"
              style={{ color: "var(--text-secondary)" }}
            >
              <Github size={12} />
              GitHub — DeepCoin-Core
            </a>
          </div>
        </div>

        {/* Bottom bar */}
        <div
          className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-6 border-t text-xs"
          style={{
            borderColor: "var(--border)",
            color:       "var(--text-muted)",
          }}
        >
          <span>© 2026 DeepCoin · ESPRIT × YEBNI · Dhia Chaieb</span>
          <a
            href="https://github.com/ChaiebDhia/DeepCoin-Core"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 hover:text-white transition-colors"
          >
            <Github size={13} />
            ChaiebDhia / DeepCoin-Core
          </a>
        </div>
      </div>
    </footer>
  );
}
