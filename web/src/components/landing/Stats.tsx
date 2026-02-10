import { motion } from "framer-motion";
import { usePrediction } from "@/hooks/usePrediction";
import { useBinanceWebSocket } from "@/hooks/useBinanceWebSocket";
import { Badge } from "@/components/ui/Badge";
import { formatPrice, formatBps, formatPercent } from "@/lib/formatters";

export function Stats() {
  const { currentCandle, isConnected } = useBinanceWebSocket();
  const { data: prediction, dataUpdatedAt } = usePrediction();

  const price = currentCandle?.close;
  const secsSinceUpdate = dataUpdatedAt
    ? Math.floor((Date.now() - dataUpdatedAt) / 1000)
    : null;

  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className="max-w-6xl mx-auto px-4 py-8"
    >
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="BTC Price" loading={!price}>
          {price ? formatPrice(price) : null}
        </StatCard>

        <StatCard label="Prediction" loading={!prediction}>
          {prediction ? <Badge direction={prediction.direction} /> : null}
        </StatCard>

        <StatCard label="Magnitude" loading={!prediction}>
          {prediction ? formatBps(prediction.magnitude_bps) : null}
        </StatCard>

        <StatCard label="Confidence" loading={!prediction}>
          {prediction?.confidence != null
            ? formatPercent(prediction.confidence)
            : null}
        </StatCard>
      </div>

      <div className="flex items-center justify-center gap-4 mt-4 text-xs text-text-secondary">
        <span className="flex items-center gap-1.5">
          <span
            className={`w-1.5 h-1.5 rounded-full ${isConnected ? "bg-up" : "bg-down"}`}
          />
          {isConnected ? "Live" : "Reconnecting..."}
        </span>
        {secsSinceUpdate != null && (
          <span>Last prediction: {secsSinceUpdate}s ago</span>
        )}
      </div>
    </motion.section>
  );
}

function StatCard({
  label,
  loading,
  children,
}: {
  label: string;
  loading: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-bg-card border border-border-subtle rounded-xl p-4 text-center">
      <div className="text-text-secondary text-xs font-medium mb-2">
        {label}
      </div>
      <div className="text-text-primary text-lg font-semibold min-h-[28px] flex items-center justify-center">
        {loading ? (
          <div className="w-16 h-5 bg-border-subtle rounded animate-pulse" />
        ) : (
          children
        )}
      </div>
    </div>
  );
}
