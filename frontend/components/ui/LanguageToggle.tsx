"use client";

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Globe } from 'lucide-react';

export function LanguageToggle() {
  const router = useRouter();
  const [locale, setLocale] = useState('en');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const match = document.cookie.match(/(?:^|; )NEXT_LOCALE=([^;]*)/);
    if (match) setLocale(match[1] || 'en');
  }, []);

  const handleSetLocale = (newLocale: string) => {
    if (newLocale === locale) return;
    // eslint-disable-next-line react-hooks/immutability
    document.cookie = "NEXT_LOCALE=" + newLocale + "; path=/; max-age=31536000";
    setLocale(newLocale);
    router.refresh(); 
  };

  if (!mounted) return null;

  return (
    <div className="flex items-center gap-1 p-0.5 rounded-full border shadow-inner transition-colors duration-300"
         style={{ background: "var(--surface-2)", borderColor: "var(--border)" }}>
      <div className="flex items-center justify-center pl-2 pr-1" style={{ color: "var(--text-muted)" }}>
        <Globe size={14} />
      </div>
      <div className="relative flex items-center">
        {['en', 'fr'].map((l) => (
          <button
            key={l}
            onClick={() => handleSetLocale(l)}
            className={`relative z-10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest transition-colors duration-200`}
            style={{ color: locale === l ? "var(--text-primary)" : "var(--text-muted)" }}
          >
            {l}
            {locale === l && (
              <motion.div
                layoutId="active-language"
                initial={false}
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                className="absolute inset-0 rounded-full -z-10 shadow-sm border"
                style={{ background: "var(--surface-1)", borderColor: "var(--border)" }}
              />
            )}
          </button>
        ))}
      </div>
    </div>
  );
}



