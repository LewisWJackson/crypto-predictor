import type { PredictionResponse } from "@/types/prediction";
import { API_URL, WAITLIST_URL } from "./constants";

export async function fetchPrediction(): Promise<PredictionResponse> {
  const res = await fetch(API_URL);
  if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);
  return res.json();
}

const GOOGLE_SHEET_URL =
  "https://script.google.com/macros/s/AKfycbyX0OGEOldyUoU4LI-HC9W5Qd3IPO9IIBk0xlawHaDAvxFZt_1ZZQZ_0oYHPH7bjoBn9g/exec";

export async function submitWaitlist(name: string, email: string) {
  // POST to Google Sheets via Apps Script
  // mode: no-cors because Apps Script redirects and doesn't set CORS headers,
  // but the request goes through and data lands in the sheet
  await fetch(GOOGLE_SHEET_URL, {
    method: "POST",
    mode: "no-cors",
    headers: { "Content-Type": "text/plain" },
    body: JSON.stringify({ name, email }),
  });

  return { success: true, message: "You're on the list." };
}
