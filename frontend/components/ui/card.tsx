/**
 * components/ui/card.tsx
 * ======================
 * Container card with optional header, content and footer slots.
 * Mirrors the bordered section style used in the PDF reports.
 */

import * as React from "react";
import { cn } from "@/lib/utils";

// ── Root ──────────────────────────────────────────────────────────────────────

const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-xl border border-[var(--border)] bg-[var(--surface-1)] text-[var(--text-primary)]",
        "shadow-lg shadow-black/30",
        className,
      )}
      {...props}
    />
  ),
);
Card.displayName = "Card";

// ── Header ────────────────────────────────────────────────────────────────────

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "flex items-center justify-between gap-3 px-5 py-4",
        "border-b border-[var(--border)]",
        className,
      )}
      {...props}
    />
  ),
);
CardHeader.displayName = "CardHeader";

// ── Title ─────────────────────────────────────────────────────────────────────

const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn("text-sm font-semibold tracking-wide text-[var(--text-primary)] uppercase", className)}
      {...props}
    />
  ),
);
CardTitle.displayName = "CardTitle";

// ── Content ───────────────────────────────────────────────────────────────────

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("px-5 py-4", className)} {...props} />
  ),
);
CardContent.displayName = "CardContent";

// ── Footer ────────────────────────────────────────────────────────────────────

const CardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "flex items-center px-5 py-3 border-t border-[var(--border)]",
        "text-xs text-[var(--text-muted)]",
        className,
      )}
      {...props}
    />
  ),
);
CardFooter.displayName = "CardFooter";

export { Card, CardHeader, CardTitle, CardContent, CardFooter };
