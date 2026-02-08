"""
Custom loss functions for TFT crypto price prediction.

Provides:
- CombinedTradingLoss: MSE + directional penalty + return-weighted MSE
- QuantileLoss wrapper for fallback usage
"""

import torch
from pytorch_forecasting.metrics import MultiHorizonMetric, QuantileLoss


class CombinedTradingLoss(MultiHorizonMetric):
    """
    Combined loss: MSE + directional penalty + return-weighted MSE.

    L = w_mse * MSE + w_dir * DirectionalPenalty + w_rw * ReturnWeightedMSE

    where:
        MSE = (y_true - y_pred)^2
        DirectionalPenalty = 1 - I(sign(y_true) == sign(y_pred))
        ReturnWeightedMSE = (y_true - y_pred)^2 * (1 + |y_true| * 100)

    Optional dead_zone: when |target| < dead_zone, the directional penalty
    is smoothly attenuated (targets near zero have ambiguous direction).

    Compatible with pytorch_forecasting's TFT via MultiHorizonMetric interface.
    """

    def __init__(
        self,
        w_mse: float = 0.3,
        w_dir: float = 0.4,
        w_rw: float = 0.3,
        dead_zone: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_mse = w_mse
        self.w_dir = w_dir
        self.w_rw = w_rw
        self.dead_zone = dead_zone

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined trading loss (per-element).

        Args:
            y_pred: Predicted values, shape (batch_size, decoder_length) or
                    (batch_size, decoder_length, 1) when output_size=1.
            target: Actual values, shape (batch_size, decoder_length)

        Returns:
            Loss tensor, shape (batch_size, decoder_length).
            pytorch-forecasting handles the reduction.
        """
        # Squeeze trailing singleton dim when output_size=1
        if y_pred.ndim == 3 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)

        # MSE component (per-element)
        mse = (target - y_pred) ** 2

        # Directional penalty (per-element)
        # 1.0 when direction is wrong, 0.0 when correct
        # Use soft approximation for gradient flow: sigmoid of product
        sign_product = target * y_pred
        # Smooth approximation: 1 - sigmoid(scale * target * y_pred)
        # When signs agree, product is positive -> sigmoid approaches 1 -> penalty ~ 0
        # When signs disagree, product is negative -> sigmoid approaches 0 -> penalty ~ 1
        direction_penalty = 1.0 - torch.sigmoid(100.0 * sign_product)

        # Dead zone: smoothly attenuate directional penalty for small moves
        # where direction is ambiguous. Uses sigmoid ramp: weight goes from
        # ~0 at |target|=0 to ~1 at |target|=2*dead_zone.
        if self.dead_zone > 0:
            # Scale factor: sigmoid reaches ~0.95 at 3*dead_zone
            dz_scale = 3.0 / (self.dead_zone + 1e-8)
            dz_weight = torch.sigmoid(dz_scale * (torch.abs(target) - self.dead_zone))
            direction_penalty = direction_penalty * dz_weight

        # Return-weighted MSE (per-element)
        # Larger actual moves get more weight
        return_weight = 1.0 + torch.abs(target) * 100.0
        return_weighted_mse = mse * return_weight

        # Combined loss (per-element)
        loss = (
            self.w_mse * mse
            + self.w_dir * direction_penalty
            + self.w_rw * return_weighted_mse
        )

        return loss

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Convert network output to point prediction.

        Handles different output sizes:
        - output_size=1: squeeze the last dimension
        - output_size=7 (quantile-style): take the median (index 3)
        - 2D output: pass through as-is
        """
        if y_pred.ndim == 3:
            if y_pred.size(-1) == 1:
                return y_pred.squeeze(-1)
            else:
                # For quantile-style output, take the median (index 3 of 7)
                return y_pred[..., y_pred.size(-1) // 2]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Convert to quantiles (pass through if already quantiles).

        For output_size=1, unsqueeze to add a quantile dim if needed.
        """
        if y_pred.ndim == 2:
            return y_pred.unsqueeze(-1)
        return y_pred


def create_quantile_loss(quantiles=None):
    """Create a QuantileLoss instance with the standard quantiles.

    This is the recommended fallback loss function for stable training.
    """
    if quantiles is None:
        quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    return QuantileLoss(quantiles=quantiles)
