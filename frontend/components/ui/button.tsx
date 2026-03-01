/**
 * components/ui/button.tsx
 * ========================
 * Reusable button primitive using class-variance-authority (CVA).
 *
 * WHY CVA for variants:
 *   CVA generates deterministic Tailwind class strings for every
 *   variant combination at compile time. No runtime style computation,
 *   no conflicts — the same mental model as shadcn/ui.
 */

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  // Base classes applied to every button
  [
    "inline-flex items-center justify-center gap-2 whitespace-nowrap",
    "rounded-md text-sm font-medium transition-colors",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
    "disabled:pointer-events-none disabled:opacity-50",
    "cursor-pointer select-none",
  ],
  {
    variants: {
      variant: {
        primary:   "bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800",
        secondary: "bg-[var(--surface-2)] text-[var(--text-primary)] border border-[var(--border)] hover:bg-[var(--surface-3)]",
        ghost:     "text-[var(--text-secondary)] hover:bg-[var(--surface-2)] hover:text-[var(--text-primary)]",
        danger:    "bg-red-600 text-white hover:bg-red-700",
        gold:      "bg-[var(--brand-gold)] text-[var(--brand-navy)] hover:opacity-90 font-semibold",
      },
      size: {
        sm:   "h-8  px-3 text-xs",
        md:   "h-10 px-4 text-sm",
        lg:   "h-11 px-6 text-base",
        icon: "h-9  w-9 p-0",
      },
    },
    defaultVariants: {
      variant: "primary",
      size:    "md",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  ),
);
Button.displayName = "Button";

export { Button, buttonVariants };
