"""
Market regime detection using Hidden Markov Models.

Identifies 4 market regimes inspired by Wyckoff market cycle theory:
- Accumulation: Low volatility, neutral returns (range-bound bottoming)
- Markup: Positive returns, moderate volatility (bull trend)
- Distribution: Low volatility, neutral-negative returns (topping)
- Markdown: Negative returns, high volatility (bear trend)

Plus 4 transition states between adjacent Wyckoff phases:
- accumulation_to_markup: Breakout — highest alpha opportunity
- markup_to_distribution: Topping — scale out, tighten stops
- distribution_to_markdown: Breakdown — exit immediately
- markdown_to_accumulation: Bottoming — start building positions

Uses a 4-state Gaussian HMM fitted on raw (pre-normalization) features:
log_return_60, rolling_volatility_20, rolling_volatility_60, volume_sma_ratio.

The RegimeTracker maintains state across prediction cycles to detect
conviction decay and trigger transition states.
"""

import pickle
from collections import deque
from dataclasses import dataclass, field
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

# Wyckoff phase labels (steady-state)
REGIME_NAMES = ["accumulation", "markup", "distribution", "markdown"]

# Valid Wyckoff cycle transitions (from -> to)
VALID_TRANSITIONS = {
    0: 1,  # accumulation -> markup
    1: 2,  # markup -> distribution
    2: 3,  # distribution -> markdown
    3: 0,  # markdown -> accumulation
}

# Transition labels
TRANSITION_NAMES = {
    (0, 1): "accumulation_to_markup",
    (1, 2): "markup_to_distribution",
    (2, 3): "distribution_to_markdown",
    (3, 0): "markdown_to_accumulation",
}

# All possible state names (4 steady + 4 transitions)
ALL_STATE_NAMES = REGIME_NAMES + [
    "accumulation_to_markup",
    "markup_to_distribution",
    "distribution_to_markdown",
    "markdown_to_accumulation",
]


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


@dataclass
class TrackerState:
    """Output from the RegimeTracker including transition detection."""

    regime: str                 # Current steady-state regime name
    regime_idx: int             # Current steady-state regime index (0-3)
    state: str                  # Effective state (regime OR transition name)
    is_transition: bool         # True if in a transition state
    transition_from: str | None # Source regime if transitioning
    transition_to: str | None   # Target regime if transitioning
    conviction: float           # Current regime conviction (0-1)
    conviction_trend: float     # Rate of conviction change (negative = decaying)
    probs: dict                 # Full regime probability dict
    successor_prob: float       # Probability of the valid Wyckoff successor


class RegimeTracker:
    """Stateful regime tracker that detects transitions via conviction decay.

    Maintains a rolling window of regime probabilities across prediction
    cycles. Detects when the current regime's conviction is weakening and
    a valid Wyckoff successor is strengthening, signaling a transition.

    Args:
        detector: A fitted RegimeDetector.
        window_size: Number of recent evaluations to track.
        conviction_threshold: Below this, the regime is "uncertain".
        decay_rate_threshold: Conviction drop per step that triggers transition.
        successor_rise_threshold: Minimum successor prob increase to confirm transition.
    """

    def __init__(
        self,
        detector: RegimeDetector,
        window_size: int = 20,
        conviction_threshold: float = 0.55,
        decay_rate_threshold: float = 0.02,
        successor_rise_threshold: float = 0.05,
    ):
        self.detector = detector
        self.window_size = window_size
        self.conviction_threshold = conviction_threshold
        self.decay_rate_threshold = decay_rate_threshold
        self.successor_rise_threshold = successor_rise_threshold

        # Rolling history of regime probabilities
        self._prob_history: deque = deque(maxlen=window_size)
        # The regime we currently believe we're in
        self._believed_regime: int | None = None
        # How many consecutive evaluations we've been in this regime
        self._regime_tenure: int = 0

    def reset(self):
        """Reset tracker state (e.g. on reconnect or data gap)."""
        self._prob_history.clear()
        self._believed_regime = None
        self._regime_tenure = 0

    def update(self, df: pd.DataFrame) -> TrackerState:
        """Run one evaluation cycle: detect regime and check for transitions.

        Args:
            df: DataFrame with at least the last row containing current features.
                Pass the full feature_df from the prediction pipeline.

        Returns:
            TrackerState with current regime, transition info, and conviction.
        """
        # Get fresh probabilities from HMM
        result = self.detector.predict(df.tail(1))
        current_probs = result.probabilities[0]  # (4,)
        current_label = int(result.labels[0])

        # Store in history
        self._prob_history.append(current_probs.copy())

        # Build probability dict
        probs_dict = {
            name: float(current_probs[i])
            for i, name in enumerate(REGIME_NAMES)
        }

        # Current conviction = probability of the dominant regime
        conviction = float(current_probs[current_label])

        # Initialize believed regime if first call
        if self._believed_regime is None:
            self._believed_regime = current_label
            self._regime_tenure = 1
        elif current_label == self._believed_regime:
            self._regime_tenure += 1
        else:
            # HMM says different regime — but don't flip immediately
            # Check if conviction supports the change
            if conviction > self.conviction_threshold and self._regime_tenure < 3:
                # Very early in previous regime and new one is strong — accept flip
                self._believed_regime = current_label
                self._regime_tenure = 1
            elif conviction > 0.7:
                # Overwhelming evidence — accept regime change
                self._believed_regime = current_label
                self._regime_tenure = 1

        # Compute conviction trend (slope of believed regime's prob over window)
        conviction_trend = self._compute_conviction_trend()

        # Check for transition state
        believed = self._believed_regime
        believed_conviction = float(current_probs[believed])
        successor_idx = VALID_TRANSITIONS.get(believed)
        successor_prob = float(current_probs[successor_idx]) if successor_idx is not None else 0.0

        is_transition = False
        transition_from = None
        transition_to = None
        effective_state = REGIME_NAMES[believed]

        if successor_idx is not None and len(self._prob_history) >= 3:
            # Transition triggers when:
            # 1. Believed regime conviction is decaying
            # 2. AND drops below threshold
            # 3. AND valid successor is rising
            conviction_decaying = conviction_trend < -self.decay_rate_threshold
            conviction_weak = believed_conviction < self.conviction_threshold
            successor_rising = self._is_rising(successor_idx)

            if conviction_weak and (conviction_decaying or successor_rising):
                is_transition = True
                transition_from = REGIME_NAMES[believed]
                transition_to = REGIME_NAMES[successor_idx]
                effective_state = TRANSITION_NAMES.get(
                    (believed, successor_idx),
                    f"{transition_from}_to_{transition_to}"
                )

            # If successor has overtaken believed regime, complete the transition
            if successor_prob > believed_conviction and successor_prob > self.conviction_threshold:
                self._believed_regime = successor_idx
                self._regime_tenure = 1
                is_transition = False
                effective_state = REGIME_NAMES[successor_idx]

        return TrackerState(
            regime=REGIME_NAMES[believed],
            regime_idx=believed,
            state=effective_state,
            is_transition=is_transition,
            transition_from=transition_from,
            transition_to=transition_to,
            conviction=believed_conviction,
            conviction_trend=conviction_trend,
            probs=probs_dict,
            successor_prob=successor_prob,
        )

    def _compute_conviction_trend(self) -> float:
        """Compute the slope of the believed regime's probability over the window."""
        if len(self._prob_history) < 3 or self._believed_regime is None:
            return 0.0

        believed = self._believed_regime
        recent = [p[believed] for p in self._prob_history]

        # Simple linear regression slope
        n = len(recent)
        x = np.arange(n, dtype=float)
        y = np.array(recent, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-8))

        return slope

    def _is_rising(self, regime_idx: int) -> bool:
        """Check if a regime's probability is trending upward."""
        if len(self._prob_history) < 3:
            return False

        recent = [p[regime_idx] for p in self._prob_history]
        # Compare last third to first third
        third = max(1, len(recent) // 3)
        early_avg = np.mean(recent[:third])
        late_avg = np.mean(recent[-third:])

        return (late_avg - early_avg) > self.successor_rise_threshold
