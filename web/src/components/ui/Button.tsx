import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary";
}

export function Button({
  variant = "primary",
  className,
  children,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(
        "px-6 py-3 rounded-lg font-semibold text-sm transition-all duration-200 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed",
        variant === "primary" &&
          "bg-accent-cyan text-bg-primary shadow-[0_0_20px_rgba(0,212,255,0.25)] hover:shadow-[0_0_40px_rgba(0,212,255,0.4)] hover:brightness-110",
        variant === "secondary" &&
          "bg-bg-card border border-border-subtle text-text-primary hover:bg-bg-hover",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
