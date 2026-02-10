export interface CandleData {
  time: number; // Unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface LinePoint {
  time: number;
  value: number;
}
