import clsx from "clsx";
import type { InputHTMLAttributes } from "react";

export function Input({
  className,
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={clsx(
        "w-full px-4 py-3 rounded-lg bg-bg-card border border-border-subtle text-text-primary placeholder:text-text-secondary/50",
        "focus:outline-none focus:border-accent-cyan/50 focus:shadow-[0_0_15px_rgba(0,212,255,0.1)]",
        "transition-all duration-200 text-sm",
        className
      )}
      {...props}
    />
  );
}
