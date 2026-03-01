/**
 * components/ui/spinner.tsx
 * =========================
 * Animated SVG spinner for loading states.
 * Sizing via size prop to avoid font-size coupling.
 */

import { cn } from "@/lib/utils";

interface SpinnerProps {
  /** Size in pixels (default 20). */
  size?:      number;
  className?: string;
}

export function Spinner({ size = 20, className }: SpinnerProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn("animate-spin text-blue-400", className)}
      aria-label="Loading"
      role="status"
    >
      {/* Three-quarter circle arc */}
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}
