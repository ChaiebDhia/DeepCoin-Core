/**
 * components/ui/footer.tsx
 * ==========================
 * Site-wide footer — Server Component.
 */

import Link from "next/link";
import { Github, ExternalLink, Linkedin, Globe, Code2 } from "lucide-react";

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
      { label: "Engineering Docs", href: "/docs" },
      { label: "Contact Us",       href: "/contact" },
    ],
  },
];

const ARCHITECTURE = [
  "Deep Learning Pipeline",
  "Multi-Agent Orchestration",
  "Hybrid RAG Engine",
  "Event-Driven Analytics",
  "Next.js App Router Server",
];

const NETWORK_LINKS = [
  { label: "Dhia Chaieb Portfolio", href: "https://dhiashayeb.vercel.app/", icon: Code2, external: true },
  { label: "LinkedIn Profile", href: "https://www.linkedin.com/in/dhia-shayeb/", icon: Linkedin, external: true },
  { label: "GitHub Repository", href: "https://github.com/ChaiebDhia/DeepCoin-Core", icon: Github, external: true },
  { label: "ESPRIT Engineering", href: "https://www.esprit.tn/", icon: Globe, external: true },
  { label: "YEBNI", href: "https://www.yebni.com/", icon: Globe, external: true },
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
              <h3 className="text-sm font-black uppercase tracking-widest mb-4" style={{ color: "var(--brand-gold)" }}>
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

          {/* Core Architecture */}
          <div>
            <h3 className="text-sm font-black uppercase tracking-widest mb-4" style={{ color: "var(--brand-gold)" }}>
              Core Architecture
            </h3>
            <ul className="space-y-2.5">
              {ARCHITECTURE.map((t) => (
                <li key={t} className="text-xs transition-colors" style={{ color: "var(--text-secondary)" }}>
                  {t}
                </li>
              ))}
            </ul>
          </div>

          {/* Network & Partners */}
          <div>
            <h3 className="text-sm font-black uppercase tracking-widest mb-4" style={{ color: "var(--brand-gold)" }}>
              Network & Partners
            </h3>
            <ul className="space-y-2.5">
              {NETWORK_LINKS.map(({ label, href, icon: Icon }) => (
                <li key={label}>
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs inline-flex items-center gap-1.5 hover:text-white transition-colors"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    <Icon size={12} className="opacity-60" />
                    {label}
                  </a>
                </li>
              ))}
            </ul>
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
          <span>© 2026 DeepCoin · PFE Internship</span>       
          <div className="flex items-center gap-1.5">
            <span>Architected By</span>
            <span className="font-semibold tracking-wide text-sm" style={{ color: "var(--brand-gold)", textShadow: "0 0 10px rgba(255, 215, 0, 0.15)" }}>
              Dhia Chaieb
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
