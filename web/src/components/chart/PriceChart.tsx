import { useEffect, useRef, useMemo, useCallback } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
  LineStyle,
} from "lightweight-charts";
import { useBinanceWebSocket } from "@/hooks/useBinanceWebSocket";
import { usePrediction } from "@/hooks/usePrediction";
import { COLORS } from "@/lib/constants";
import type { CandleData } from "@/types/chart";

const ts = (n: number) => n as UTCTimestamp;

const PREDICTION_STEPS = 30; // predict 30 minutes ahead for more visual impact

/**
 * Generate a simulated prediction from recent price action.
 * Each step has its own distinct movement to avoid a straight line.
 * Returns per-step individual returns (not cumulative).
 */
function simulateStepReturns(candles: CandleData[]): number[] {
  if (candles.length < 30) return [];

  const recent = candles.slice(-30);

  // Gather per-bar returns
  const returns: number[] = [];
  for (let i = 1; i < recent.length; i++) {
    returns.push(Math.log(recent[i].close / recent[i - 1].close));
  }

  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const std = Math.sqrt(
    returns.reduce((s, r) => s + (r - avgReturn) ** 2, 0) / (returns.length - 1)
  );

  // Use recent candle patterns to seed realistic per-step returns
  const steps: number[] = [];
  for (let i = 0; i < PREDICTION_STEPS; i++) {
    // Sample from recent actual returns (wrapping), scaled by damping
    const sampleIdx = i % returns.length;
    const sampleReturn = returns[sampleIdx];

    // Damped momentum — trend fades further out
    const dampFactor = Math.max(0.1, 1 - i / (PREDICTION_STEPS * 1.5));

    // Mix: 60% sampled pattern + 40% trend + noise
    const trend = avgReturn * dampFactor;
    const pattern = sampleReturn * dampFactor * 0.6;
    const noise = std * 0.4 * Math.sin(i * 3.7 + 1.3) * Math.cos(i * 1.1 + 0.5);

    steps.push(trend * 0.4 + pattern + noise);
  }

  return steps;
}

/**
 * Convert per-step returns into cumulative price points for the prediction line.
 */
function buildPredictionLine(
  anchorPrice: number,
  anchorTime: number,
  stepReturns: number[]
): { time: number; value: number }[] {
  const points: { time: number; value: number }[] = [];
  let cumReturn = 0;

  for (let i = 0; i < stepReturns.length; i++) {
    cumReturn += stepReturns[i];
    points.push({
      time: anchorTime + (i + 1) * 60,
      value: anchorPrice * Math.exp(cumReturn),
    });
  }

  return points;
}

export function PriceChart() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const predLineRef = useRef<ISeriesApi<"Line"> | null>(null);
  const bandUpperRef = useRef<ISeriesApi<"Area"> | null>(null);
  const bandLowerRef = useRef<ISeriesApi<"Area"> | null>(null);
  const prevCandleCountRef = useRef(0);
  const hasSetInitialRange = useRef(false);

  // Persistent prediction: stores locked prediction points that stay on chart
  const lockedPointsRef = useRef<{ time: number; value: number }[]>([]);
  const lastPredictionAnchorRef = useRef<number>(0);

  const { candles, currentCandle, isConnected } = useBinanceWebSocket();
  const { data: prediction } = usePrediction();

  // Compute recent volatility from last 30 candles
  const recentVol = useMemo(() => {
    if (candles.length < 10) return 0.001;
    const recent = candles.slice(-30);
    const ranges = recent.map((c) => (c.high - c.low) / c.close);
    return ranges.reduce((a, b) => a + b, 0) / ranges.length;
  }, [candles]);

  // Simulated step returns when API is unavailable
  const simStepReturns = useMemo(() => {
    if (prediction) return null;
    return simulateStepReturns(candles);
  }, [candles, prediction]);

  const isSimulated = !prediction && !!simStepReturns?.length;
  const hasPrediction = !!(prediction?.predictions_all_steps?.length || simStepReturns?.length);

  // Scroll to show the prediction area
  const scrollToShowPrediction = useCallback((anchorTime: number) => {
    if (!chartRef.current) return;
    const fromTime = anchorTime - 45 * 60; // 45 min of real history
    const toTime = anchorTime + (PREDICTION_STEPS + 3) * 60; // all prediction steps + padding
    chartRef.current.timeScale().setVisibleRange({
      from: ts(fromTime),
      to: ts(toTime),
    });
  }, []);

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: COLORS.bgPrimary },
        textColor: COLORS.textSecondary,
        fontFamily: "Inter, system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: "rgba(30, 30, 48, 0.5)" },
        horzLines: { color: "rgba(30, 30, 48, 0.5)" },
      },
      crosshair: {
        vertLine: { color: "rgba(0, 212, 255, 0.3)", labelBackgroundColor: COLORS.accentCyan },
        horzLine: { color: "rgba(0, 212, 255, 0.3)", labelBackgroundColor: COLORS.accentCyan },
      },
      rightPriceScale: { borderColor: COLORS.borderSubtle },
      timeScale: {
        borderColor: COLORS.borderSubtle,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
      },
    });

    chartRef.current = chart;

    // Real candlestick series
    candleSeriesRef.current = chart.addCandlestickSeries({
      upColor: COLORS.up,
      downColor: COLORS.down,
      borderVisible: false,
      wickUpColor: COLORS.up,
      wickDownColor: COLORS.down,
    });

    // Confidence band — upper
    bandUpperRef.current = chart.addAreaSeries({
      topColor: "rgba(0, 180, 255, 0.08)",
      bottomColor: "rgba(0, 180, 255, 0.08)",
      lineColor: "rgba(0, 180, 255, 0.15)",
      lineWidth: 1,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // Confidence band — lower eraser
    bandLowerRef.current = chart.addAreaSeries({
      topColor: COLORS.bgPrimary,
      bottomColor: COLORS.bgPrimary,
      lineColor: "rgba(0, 180, 255, 0.15)",
      lineWidth: 1,
      crosshairMarkerVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
    });

    // Prediction line — blue neon
    predLineRef.current = chart.addLineSeries({
      color: "#00d4ff",
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 4,
      crosshairMarkerBackgroundColor: "#00d4ff",
      lastValueVisible: true,
      priceLineVisible: false,
    });

    const ro = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, []);

  // Update real candle data
  useEffect(() => {
    if (!candleSeriesRef.current || candles.length === 0) return;

    const allCandles: CandleData[] = [...candles];
    if (currentCandle) {
      const lastIdx = allCandles.findIndex((c) => c.time === currentCandle.time);
      if (lastIdx >= 0) {
        allCandles[lastIdx] = currentCandle;
      } else {
        allCandles.push(currentCandle);
      }
    }

    if (prevCandleCountRef.current !== candles.length) {
      candleSeriesRef.current.setData(
        allCandles.map((c) => ({ ...c, time: ts(c.time) }))
      );
      prevCandleCountRef.current = candles.length;
    } else if (currentCandle) {
      candleSeriesRef.current.update({ ...currentCandle, time: ts(currentCandle.time) });
    }
  }, [candles, currentCandle]);

  // Update prediction line — re-anchors on every tick
  useEffect(() => {
    if (!predLineRef.current || !currentCandle) return;

    const anchorPrice = currentCandle.close;
    const anchorTime = currentCandle.time;

    // When a new candle opens (anchor time changes), lock the previous prediction's
    // first point and trim old locked points that real time has passed
    if (anchorTime !== lastPredictionAnchorRef.current && lastPredictionAnchorRef.current > 0) {
      // Remove any locked points that real candles have now passed
      lockedPointsRef.current = lockedPointsRef.current.filter(
        (p) => p.time > anchorTime
      );
    }
    lastPredictionAnchorRef.current = anchorTime;

    let linePoints: { time: number; value: number }[];

    if (prediction?.predictions_all_steps) {
      // Real model predictions (cumulative returns already)
      linePoints = prediction.predictions_all_steps.map((r, i) => ({
        time: anchorTime + (i + 1) * 60,
        value: anchorPrice * Math.exp(r),
      }));
    } else if (simStepReturns?.length) {
      // Simulated predictions
      linePoints = buildPredictionLine(anchorPrice, anchorTime, simStepReturns);
    } else {
      return;
    }

    // Bridge point: start from current price
    const bridge = { time: anchorTime, value: anchorPrice };

    // Merge locked historical predictions with live predictions
    const merged = [
      bridge,
      ...lockedPointsRef.current.filter((p) => p.time > anchorTime && p.time <= linePoints[0]?.time),
      ...linePoints,
    ];

    // Sort by time and deduplicate
    const seen = new Set<number>();
    const final = merged
      .sort((a, b) => a.time - b.time)
      .filter((p) => {
        if (seen.has(p.time)) return false;
        seen.add(p.time);
        return true;
      });

    predLineRef.current.setData(
      final.map((p) => ({ time: ts(p.time), value: p.value }))
    );

    // Confidence bands
    if (prediction?.quantiles_all_steps && bandUpperRef.current && bandLowerRef.current) {
      const b = { time: ts(anchorTime), value: anchorPrice };
      const upper = [b, ...prediction.quantiles_all_steps.map((q, i) => ({
        time: ts(anchorTime + (i + 1) * 60),
        value: anchorPrice * Math.exp(q.p90),
      }))];
      const lower = [b, ...prediction.quantiles_all_steps.map((q, i) => ({
        time: ts(anchorTime + (i + 1) * 60),
        value: anchorPrice * Math.exp(q.p10),
      }))];
      bandUpperRef.current.setData(upper);
      bandLowerRef.current.setData(lower);
    } else if (isSimulated && bandUpperRef.current && bandLowerRef.current) {
      const b = { time: ts(anchorTime), value: anchorPrice };
      const vol = recentVol * 0.7;

      // Build bands from the line points for more realistic shape
      const upper = [b, ...linePoints.map((p, i) => ({
        time: ts(p.time),
        value: p.value * (1 + vol * (i + 1) * 0.12),
      }))];
      const lower = [b, ...linePoints.map((p, i) => ({
        time: ts(p.time),
        value: p.value * (1 - vol * (i + 1) * 0.12),
      }))];

      bandUpperRef.current.setData(upper);
      bandLowerRef.current.setData(lower);
    }

    // Set initial visible range to show predictions
    if (!hasSetInitialRange.current && candles.length > 0) {
      hasSetInitialRange.current = true;
      scrollToShowPrediction(anchorTime);
    }
  }, [prediction, simStepReturns, currentCandle, recentVol, isSimulated, candles.length, scrollToShowPrediction]);

  return (
    <div className="relative">
      <div
        ref={containerRef}
        className="w-full h-[500px] md:h-[600px] rounded-xl overflow-hidden border border-border-subtle"
      />

      {/* Chart legend */}
      <div className="absolute top-4 left-4 flex flex-col gap-2 pointer-events-none z-10">
        <div className="flex items-center gap-2 bg-bg-primary/80 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-border-subtle">
          <span className="w-3 h-3 rounded-sm bg-up" />
          <span className="text-xs text-text-secondary">Live BTC/USDT</span>
          {isConnected && (
            <span className="w-1.5 h-1.5 rounded-full bg-up animate-pulse ml-1" />
          )}
        </div>

        <div className="flex items-center gap-2 bg-bg-primary/80 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-accent-cyan/20">
          <span
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: "#00d4ff", boxShadow: "0 0 6px #00d4ff" }}
          />
          <span className="text-xs text-accent-cyan">
            {isSimulated
              ? `AI Preview — next ${PREDICTION_STEPS} min`
              : `AI Prediction — next ${PREDICTION_STEPS} min`}
          </span>
          {!isSimulated && prediction && (
            <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse ml-1" />
          )}
        </div>
      </div>

      {/* Prediction disclaimer */}
      {hasPrediction && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 pointer-events-none z-10">
          <div className="bg-bg-primary/80 backdrop-blur-sm rounded-full px-4 py-1.5 border border-accent-cyan/10">
            <span className="text-[11px] text-text-secondary">
              <span className="text-accent-cyan font-medium">Blue line</span>
              {isSimulated
                ? " is a preview — real AI predictions activate when the model is ready"
                : " shows AI-predicted prices — these haven't occurred yet"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
