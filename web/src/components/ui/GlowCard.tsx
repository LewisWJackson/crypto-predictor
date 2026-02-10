import clsx from "clsx";
import type { ReactNode } from "react";

interface GlowCardProps {
  children: ReactNode;
  className?: string;
}

export function GlowCard({ children, className }: GlowCardProps) {
  return (
    <div
      className={clsx(
        "bg-bg-card border border-border-subtle rounded-xl p-6",
        "hover:shadow-[0_0_40px_rgba(0,212,255,0.08)] hover:border-accent-cyan/20",
        "transition-all duration-300",
        className
      )}
    >
      {children}
    </div>
  );
}
