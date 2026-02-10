export const API_URL = "/predict";
export const WAITLIST_URL = "/api/waitlist";

export const BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m";
export const BINANCE_REST_KLINES = "https://api.binance.com/api/v3/klines";

export const COLORS = {
  bgPrimary: "#0a0a0f",
  bgCard: "#12121a",
  borderSubtle: "#1e1e30",
  textPrimary: "#e8e8f0",
  textSecondary: "#8888aa",
  accentCyan: "#00d4ff",
  up: "#22c55e",
  down: "#ef4444",
  predictionLine: "rgba(0, 212, 255, 0.8)",
  bandFill: "rgba(0, 212, 255, 0.12)",
  bandLine: "rgba(0, 212, 255, 0.3)",
} as const;

export const PREDICTION_POLL_INTERVAL = 60_000;
