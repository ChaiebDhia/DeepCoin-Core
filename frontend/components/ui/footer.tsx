/**
 * components/ui/footer.tsx
 * ==========================
 * Site-wide footer — Server Component.
 */

import Link from "next/link";
import { Github, ExternalLink, Linkedin, Globe, Code2 } from "lucide-react";
import { useTranslations } from "next-intl";







export function Footer() {
  const t = useTranslations("Footer");

  const COLUMNS = [
    {
      heading: t("col_product"),
      links: [
        { label: t("link_analyse"),   href: "/#analyse" },
        { label: t("link_how"),     href: "/#how-it-works" },
        { label: t("link_features"),         href: "/#features" },
      ],
    },
    {
      heading: t("col_resources"),
      links: [
        { label: t("link_cn"),  href: "https://www.corpus-nummorum.eu",  external: true },
        { label: t("link_docs"), href: "/docs" },
        { label: t("link_contact"),       href: "/contact" },
      ],
    },
  ];

  const ARCHITECTURE = [
    t("arch_1"),
    t("arch_2"),
    t("arch_3"),
    t("arch_4"),
    t("arch_5"),
  ];

  const NETWORK_LINKS = [
    { label: t("link_portfolio"), href: "https://dhiashayeb.vercel.app/", icon: Code2, external: true },
    { label: t("link_linkedin"), href: "https://www.linkedin.com/in/dhia-shayeb/", icon: Linkedin, external: true },
    { label: t("link_github"), href: "https://github.com/ChaiebDhia/DeepCoin-Core", icon: Github, external: true },
    { label: t("link_esprit"), href: "https://www.esprit.tn/", icon: Globe, external: true },
    { label: t("link_yebni"), href: "https://www.yebni.com/", icon: Globe, external: true },
  ];

  return (
    <footer
      className="border-t mt-8"
      style={{ backgroundColor: "var(--footer-bg)", color: "var(--footer-text)", borderColor: "rgba(255,255,255,0.1)" }}
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
                        style={{ color: "var(--footer-text)" }}
                      >
                        {label}
                        <ExternalLink size={10} className="opacity-60" />       
                      </a>
                    ) : (
                      <Link
                        href={href}
                        className="text-xs hover:text-white transition-colors"  
                        style={{ color: "var(--footer-text)" }}
                      >
                        {label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}

          {/* {t("col_arch")} */}
          <div>
            <h3 className="text-sm font-black uppercase tracking-widest mb-4" style={{ color: "var(--brand-gold)" }}>
              {t("col_arch")}
            </h3>
            <ul className="space-y-2.5">
              {ARCHITECTURE.map((t) => (
                <li key={t} className="text-xs transition-colors" style={{ color: "var(--footer-text)" }}>
                  {t}
                </li>
              ))}
            </ul>
          </div>

          {/* {t("col_network")} */}
          <div>
            <h3 className="text-sm font-black uppercase tracking-widest mb-4" style={{ color: "var(--brand-gold)" }}>
              {t("col_network")}
            </h3>
            <ul className="space-y-2.5">
              {NETWORK_LINKS.map(({ label, href, icon: Icon }) => (
                <li key={label}>
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs inline-flex items-center gap-1.5 hover:text-white transition-colors"
                    style={{ color: "var(--footer-text)" }}
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
            borderColor: "rgba(255,255,255,0.1)",
            color:       "var(--text-muted)",
          }}
        >
          <span>{t("copyright")}</span>       
          <div className="flex items-center gap-1.5">
            <span>{t("architected")}</span>
            <span className="font-semibold tracking-wide text-sm" style={{ color: "var(--brand-gold)", textShadow: "0 0 10px rgba(255, 215, 0, 0.15)" }}>
              Dhia Chaieb
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}



