export function formatPrice(price: number): string {
  return price.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export function formatBps(bps: number): string {
  return `${bps.toFixed(1)} bps`;
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
