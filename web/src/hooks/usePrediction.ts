import { useQuery } from "@tanstack/react-query";
import { fetchPrediction } from "@/lib/api";
import { PREDICTION_POLL_INTERVAL } from "@/lib/constants";

export function usePrediction() {
  return useQuery({
    queryKey: ["prediction"],
    queryFn: fetchPrediction,
    refetchInterval: PREDICTION_POLL_INTERVAL,
    staleTime: PREDICTION_POLL_INTERVAL - 5000,
    retry: 1,
  });
}
