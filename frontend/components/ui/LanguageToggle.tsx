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
    <div className="flex items-center gap-2 p-1 rounded-full bg-zinc-900 border border-zinc-800 shadow-inner">
      <div className="flex items-center justify-center pl-2 pr-1 text-zinc-500">
        <Globe size={14} />
      </div>
      <div className="relative flex items-center">
        {['en', 'fr'].map((l) => (
          <button
            key={l}
            onClick={() => handleSetLocale(l)}
            className={"relative z-10 px-3 py-1 text-[10px] font-semibold uppercase tracking-widest transition-colors duration-200 " + (locale === l ? 'text-white' : 'text-zinc-500 hover:text-zinc-300')}
          >
            {l}
            {locale === l && (
              <motion.div
                layoutId="active-language"
                initial={false}
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                className="absolute inset-0 bg-zinc-700/50 rounded-full -z-10 border border-zinc-600 shadow-sm"
              />
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

