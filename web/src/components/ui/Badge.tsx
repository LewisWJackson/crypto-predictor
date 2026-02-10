import clsx from "clsx";

interface BadgeProps {
  direction: "UP" | "DOWN";
}

export function Badge({ direction }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-bold tracking-wide",
        direction === "UP"
          ? "bg-up/15 text-up border border-up/30"
          : "bg-down/15 text-down border border-down/30"
      )}
    >
      {direction === "UP" ? "\u2191" : "\u2193"} {direction}
    </span>
  );
}
