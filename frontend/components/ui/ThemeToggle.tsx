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



  const iconColor = isDark ? "#fbbf24" : "#1f2937"; // Gold in dark mode, dark gray in light mode

  return (

    <button

      onClick={() => setTheme(isDark ? "light" : "dark")}

      className="relative flex items-center justify-center w-8 h-8 rounded-md transition-colors hover:bg-white/10"

      style={{ color: iconColor, opacity: 0.9 }}

      aria-label="Toggle theme"

    >

      {isDark ? <Sun size={18} /> : <Moon size={18} />}

    </button>

  );

}



