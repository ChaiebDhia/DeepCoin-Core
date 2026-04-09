import { ReactNode } from "react";

export function ThemeAwareCard({ children, className = "" }: { children: ReactNode, className?: string }) {
  return (
    <div
      className={"rounded-2xl transition-all duration-300 transform-gpu hover:-translate-y-1 overflow-hidden " + className}
      style={{
        backgroundColor: "var(--surface-1)",
        borderColor: "var(--border)",
        borderWidth: "1px",
        borderStyle: "solid",
        boxShadow: "0 4px 6px -1px var(--shadow-sm), 0 2px 4px -2px var(--shadow-sm)"
      }}
    >
      {children}
    </div>
  );
}