export interface StepQuantiles {
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
}

export interface PredictionResponse {
  direction: "UP" | "DOWN";
  predicted_return: number;
  magnitude_bps: number;
  confidence: number | null;
  horizon_minutes: number;
  predictions_all_steps: number[];
  current_price: number;
  current_timestamp: number;
  quantiles?: {
    p02: number;
    p10: number;
    p25: number;
    p50: number;
    p75: number;
    p90: number;
    p98: number;
  };
  quantiles_all_steps?: StepQuantiles[];
  regime?: "accumulation" | "markup" | "distribution" | "markdown";
  regime_probs?: {
    accumulation: number;
    markup: number;
    distribution: number;
    markdown: number;
  };
}
