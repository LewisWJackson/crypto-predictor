import { useEffect, useRef, useState, useCallback } from "react";
import type { CandleData } from "@/types/chart";
import { BINANCE_WS_URL, BINANCE_REST_KLINES } from "@/lib/constants";

const MAX_CANDLES = 500;

export function useBinanceWebSocket() {
  const [candles, setCandles] = useState<CandleData[]>([]);
  const [currentCandle, setCurrentCandle] = useState<CandleData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Seed initial candles from REST API
  const seedCandles = useCallback(async () => {
    try {
      const res = await fetch(
        `${BINANCE_REST_KLINES}?symbol=BTCUSDT&interval=1m&limit=300`
      );
      const data = await res.json();
      const parsed: CandleData[] = data.map((k: unknown[]) => ({
        time: Math.floor((k[0] as number) / 1000),
        open: parseFloat(k[1] as string),
        high: parseFloat(k[2] as string),
        low: parseFloat(k[3] as string),
        close: parseFloat(k[4] as string),
      }));
      setCandles(parsed);
    } catch (err) {
      console.error("Failed to seed candles:", err);
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(BINANCE_WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => {
      setIsConnected(false);
      reconnectTimer.current = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      const k = msg.k;
      if (!k) return;

      const candle: CandleData = {
        time: Math.floor(k.t / 1000),
        open: parseFloat(k.o),
        high: parseFloat(k.h),
        low: parseFloat(k.l),
        close: parseFloat(k.c),
      };

      setCurrentCandle(candle);

      if (k.x) {
        // Candle closed — push to history
        setCandles((prev) => {
          const updated = [...prev];
          const lastIdx = updated.findIndex((c) => c.time === candle.time);
          if (lastIdx >= 0) {
            updated[lastIdx] = candle;
          } else {
            updated.push(candle);
          }
          return updated.slice(-MAX_CANDLES);
        });
      }
    };
  }, []);

  useEffect(() => {
    seedCandles().then(connect);
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [seedCandles, connect]);

  return { candles, currentCandle, isConnected };
}
