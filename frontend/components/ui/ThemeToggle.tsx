"use client";



import { Moon, Sun } from "lucide-react";

import { useTheme } from "next-themes";

import { useEffect, useState } from "react";



export function ThemeToggle() {

  const { setTheme, theme, resolvedTheme } = useTheme();

  const [mounted, setMounted] = useState(false);



  useEffect(() => {

    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);

  }, []);



  if (!mounted) {

    return <div className="w-8 h-8 rounded-md" />;

  }



  const isDark = theme === "dark" || resolvedTheme === "dark";

  // Since the navbar and mobile dropdown are now context-aware (light in light mode, dark in dark mode),
  // we use a deep slate color for light mode and gold for dark mode.
  const iconColor = isDark ? "#fbbf24" : "#0f172a"; 
  return (
    <button
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="relative flex items-center justify-center w-8 h-8 rounded-md transition-colors hover:bg-[var(--surface-2)]"
      style={{ color: iconColor, opacity: 1 }}
      aria-label="Toggle theme"
    >
      {isDark ? <Sun size={20} /> : <Moon size={20} />}
    </button>
  );

}



