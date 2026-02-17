"""
Market regime detection using Hidden Markov Models.

Identifies 4 market regimes inspired by Wyckoff market cycle theory:
- Accumulation: Low volatility, neutral returns (range-bound bottoming)
- Markup: Positive returns, moderate volatility (bull trend)
- Distribution: Low volatility, neutral-negative returns (topping)
- Markdown: Negative returns, high volatility (bear trend)

Uses a 4-state Gaussian HMM fitted on raw (pre-normalization) features:
log_return_60, rolling_volatility_20, rolling_volatility_60, volume_sma_ratio.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# Features consumed by the HMM (raw, pre-normalization)
HMM_FEATURES = [
    "log_return_60",
    "rolling_volatility_20",
    "rolling_volatility_60",
    "volume_sma_ratio",
]

# Wyckoff phase labels
REGIME_NAMES = ["accumulation", "markup", "distribution", "markdown"]


@dataclass
class RegimeLabels:
    """Container for regime detection output."""

    labels: np.ndarray          # (N,) int array of regime indices 0-3
    probabilities: np.ndarray   # (N, 4) posterior probabilities
    names: list                 # mapping index -> regime name


class RegimeDetector:
    """4-state Gaussian HMM for market regime detection."""

    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        n_iter: int = 200,
        min_duration: int = 30,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.min_duration = min_duration
        self.random_state = random_state

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=False,
        )

        # Mapping from HMM state index -> Wyckoff phase index
        self._state_map: dict | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract HMM input features from a DataFrame."""
        missing = [f for f in HMM_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing HMM features: {missing}")
        X = df[HMM_FEATURES].values.astype(np.float64)
        # Replace any remaining NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Fit the HMM on training data.

        Args:
            df: Training DataFrame with raw (pre-normalization) features.

        Returns:
            self
        """
        X = self._extract_features(df)
        self.model.fit(X)
        self._fitted = True

        # Label states by inspecting learned means
        self._label_states()

        return self

    def _label_states(self):
        """Map HMM state indices to Wyckoff phases based on learned parameters.

        Strategy: rank states by mean return (log_return_60) and mean volatility
        (rolling_volatility_20). The mapping is:
        - Markup:       highest mean return
        - Markdown:     lowest mean return
        - Distribution: among the remaining two, higher volatility
        - Accumulation: among the remaining two, lower volatility
        """
        means = self.model.means_  # (n_states, n_features)
        ret_idx = HMM_FEATURES.index("log_return_60")
        vol_idx = HMM_FEATURES.index("rolling_volatility_20")

        mean_returns = means[:, ret_idx]
        mean_vols = means[:, vol_idx]

        # Assign markup (highest return) and markdown (lowest return)
        sorted_by_return = np.argsort(mean_returns)
        markdown_state = sorted_by_return[0]
        markup_state = sorted_by_return[-1]

        # Remaining two states
        remaining = [s for s in range(self.n_states)
                     if s not in (markup_state, markdown_state)]

        # Among remaining: higher vol = distribution, lower vol = accumulation
        if mean_vols[remaining[0]] > mean_vols[remaining[1]]:
            distribution_state = remaining[0]
            accumulation_state = remaining[1]
        else:
            distribution_state = remaining[1]
            accumulation_state = remaining[0]

        # Build map: hmm_state -> wyckoff_index
        self._state_map = {
            accumulation_state: 0,  # accumulation
            markup_state: 1,        # markup
            distribution_state: 2,  # distribution
            markdown_state: 3,      # markdown
        }

    def predict(self, df: pd.DataFrame) -> RegimeLabels:
        """Predict regime labels and probabilities using forward algorithm only.

        Uses predict_proba (forward algorithm) to avoid future information
        leakage — does NOT use Viterbi smoothing.

        Args:
            df: DataFrame with raw features.

        Returns:
            RegimeLabels with mapped labels and probabilities.
        """
        if not self._fitted:
            raise RuntimeError("RegimeDetector not fitted. Call fit() first.")

        X = self._extract_features(df)

        # Forward-only probabilities (no future info)
        raw_probs = self.model.predict_proba(X)  # (N, n_states)
        raw_labels = np.argmax(raw_probs, axis=1)

        # Apply minimum duration smoothing
        smoothed_labels = self._smooth_labels(raw_labels)

        # Remap HMM states to Wyckoff phases
        mapped_labels = np.array([self._state_map[s] for s in smoothed_labels])

        # Remap probability columns to Wyckoff order
        mapped_probs = np.zeros_like(raw_probs)
        for hmm_state, wyckoff_idx in self._state_map.items():
            mapped_probs[:, wyckoff_idx] = raw_probs[:, hmm_state]

        return RegimeLabels(
            labels=mapped_labels,
            probabilities=mapped_probs,
            names=REGIME_NAMES,
        )

    def _smooth_labels(self, labels: np.ndarray) -> np.ndarray:
        """Remove regime flips shorter than min_duration bars.

        Short-lived state changes are replaced with the surrounding regime.
        """
        if self.min_duration <= 1:
            return labels

        smoothed = labels.copy()
        n = len(labels)
        i = 0
        while i < n:
            # Find the end of the current run
            j = i + 1
            while j < n and labels[j] == labels[i]:
                j += 1
            run_length = j - i

            # If this run is too short and not at the boundaries, replace it
            if run_length < self.min_duration and i > 0:
                smoothed[i:j] = smoothed[i - 1]

            i = j

        return smoothed

    def get_state_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptive statistics per regime.

        Returns a DataFrame with mean return, mean vol, and bar count per regime.
        """
        result = self.predict(df)
        X = self._extract_features(df)
        ret_idx = HMM_FEATURES.index("log_return_60")
        vol_idx = HMM_FEATURES.index("rolling_volatility_20")

        rows = []
        for idx, name in enumerate(REGIME_NAMES):
            mask = result.labels == idx
            count = int(np.sum(mask))
            if count > 0:
                mean_ret = float(np.mean(X[mask, ret_idx]))
                mean_vol = float(np.mean(X[mask, vol_idx]))
            else:
                mean_ret = 0.0
                mean_vol = 0.0
            rows.append({
                "regime": name,
                "index": idx,
                "count": count,
                "fraction": count / len(X),
                "mean_return_60": mean_ret,
                "mean_volatility_20": mean_vol,
            })

        return pd.DataFrame(rows)

    def save(self, path: str | Path):
        """Save the fitted detector to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "state_map": self._state_map,
            "fitted": self._fitted,
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "min_duration": self.min_duration,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeDetector":
        """Load a fitted detector from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        detector = cls(
            n_states=state["n_states"],
            covariance_type=state["covariance_type"],
            min_duration=state["min_duration"],
        )
        detector.model = state["model"]
        detector._state_map = state["state_map"]
        detector._fitted = state["fitted"]
        return detector
