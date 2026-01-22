"""Tests for BagOfQuantiles ensemble with canonical aggregation.

CANONICAL AGGREGATION (Step 4):
- Each model outputs predicted τ-quantile: q̂_τ^(b)(x)
- Ensemble: q̂_τ^ens(x) = mean(q̂_τ^(b)(x))
- Margin: m(x) = q̂_τ^ens(x) - s(x)
- Decision: δ(x) = 1{m(x) > 0}

All metrics should use margin m(x) as the attack score.
"""

import torch

from attacks.qr.bagging import BagOfQuantiles


class ConstantModel(torch.nn.Module):
    """Mock model that always outputs a constant value (in log space)."""
    
    def __init__(self, value: float) -> None:
        super().__init__()
        # value is in log1p space
        self.value = torch.nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, x):  # type: ignore[override]
        batch = x.shape[0]
        return self.value.expand(batch)


def test_predict_quantile():
    """Test ensemble quantile prediction (mean aggregation)."""
    bag = BagOfQuantiles(
        base_cfg={"batch_size": 1, "epochs": 1, "lr": 1e-3, "val_ratio": 0.1, "tau_values": [0.1]},
        B=3
    )
    # Set up models with known predictions (in log space)
    # Models predict log1p values: 0.3, 0.4, 0.5
    bag.models_by_tau[0.1] = [ConstantModel(0.3), ConstantModel(0.4), ConstantModel(0.5)]
    
    imgs = torch.zeros(2, 3, 32, 32)
    q_ens, diagnostics = bag.predict_quantile(imgs, tau=0.1)
    
    # Check ensemble prediction is mean of individual predictions
    expected_q_ens = torch.tensor([0.4, 0.4])  # mean([0.3, 0.4, 0.5])
    assert torch.allclose(q_ens, expected_q_ens), f"Expected {expected_q_ens}, got {q_ens}"
    
    # Check per-model predictions are stored
    assert diagnostics["preds_per_model_log"].shape == (3, 2)
    assert diagnostics["q_ens_log"].shape == (2,)


def test_compute_margin():
    """Test canonical margin computation: m(x) = q̂_τ^ens(x) - s(x)."""
    bag = BagOfQuantiles(
        base_cfg={"batch_size": 1, "epochs": 1, "lr": 1e-3, "val_ratio": 0.1, "tau_values": [0.1]},
        B=3
    )
    # Models predict log1p values: 0.3, 0.4, 0.5 => mean = 0.4
    bag.models_by_tau[0.1] = [ConstantModel(0.3), ConstantModel(0.4), ConstantModel(0.5)]
    
    # Observed scores (raw, not log)
    # log1p(0.5) ≈ 0.405, log1p(0.7) ≈ 0.531
    scores = torch.tensor([0.5, 0.7])
    imgs = torch.zeros(2, 3, 32, 32)
    
    margin, diagnostics = bag.compute_margin(scores, imgs, tau=0.1)
    
    # Margin = q_ens (0.4) - s(x) (log1p(scores))
    expected_scores_log = torch.log1p(scores)
    expected_margin = torch.tensor([0.4, 0.4]) - expected_scores_log
    
    assert torch.allclose(margin, expected_margin, atol=1e-5), f"Expected {expected_margin}, got {margin}"
    assert torch.allclose(diagnostics["scores_log"], expected_scores_log)


def test_decision_canonical():
    """Test canonical decision rule: δ(x) = 1{m(x) > 0}."""
    bag = BagOfQuantiles(
        base_cfg={"batch_size": 1, "epochs": 1, "lr": 1e-3, "val_ratio": 0.1, "tau_values": [0.1]},
        B=3
    )
    # Models predict log1p values: 0.3, 0.4, 0.5 => mean = 0.4
    bag.models_by_tau[0.1] = [ConstantModel(0.3), ConstantModel(0.4), ConstantModel(0.5)]
    
    # Observed scores (raw)
    # log1p(0.3) ≈ 0.262 < 0.4 => margin > 0 => member
    # log1p(0.7) ≈ 0.531 > 0.4 => margin < 0 => non-member
    scores = torch.tensor([0.3, 0.7])
    imgs = torch.zeros(2, 3, 32, 32)
    
    decisions, diagnostics = bag.decision(scores, imgs, tau=0.1)
    
    # Check decisions based on margin sign
    # margin[0] = 0.4 - log1p(0.3) = 0.4 - 0.262 ≈ 0.138 > 0 => member (1)
    # margin[1] = 0.4 - log1p(0.7) = 0.4 - 0.531 ≈ -0.131 < 0 => non-member (0)
    assert decisions[0].item() == 1, f"Expected member for score=0.3, got {decisions[0].item()}"
    assert decisions[1].item() == 0, f"Expected non-member for score=0.7, got {decisions[1].item()}"
    
    # Check margin is stored in diagnostics
    assert "margin" in diagnostics
    assert diagnostics["margin"][0] > 0  # First sample should have positive margin
    assert diagnostics["margin"][1] < 0  # Second sample should have negative margin


def test_backward_compatibility_fields():
    """Test that backward compatibility fields are present in diagnostics."""
    bag = BagOfQuantiles(
        base_cfg={"batch_size": 1, "epochs": 1, "lr": 1e-3, "val_ratio": 0.1, "tau_values": [0.1]},
        B=3
    )
    bag.models_by_tau[0.1] = [ConstantModel(0.4), ConstantModel(0.4), ConstantModel(0.4)]
    
    scores = torch.tensor([0.5])
    imgs = torch.zeros(1, 3, 32, 32)
    
    decisions, diagnostics = bag.decision(scores, imgs, tau=0.1)
    
    # Check backward compatibility fields are present
    assert "thresholds_log" in diagnostics
    assert "thresholds_raw" in diagnostics
    assert "scores_log" in diagnostics
    assert "scores_raw" in diagnostics
    assert "margin" in diagnostics
    assert "decisions" in diagnostics
    
    # Check thresholds map to predictions
    assert torch.allclose(diagnostics["thresholds_log"], diagnostics["preds_per_model_log"])


def test_margin_as_attack_score():
    """Test that margin is correctly computed for use as attack score.
    
    For MIA evaluation:
    - Members should have higher margins (score below predicted quantile)
    - Non-members should have lower margins (score at or above predicted)
    """
    bag = BagOfQuantiles(
        base_cfg={"batch_size": 1, "epochs": 1, "lr": 1e-3, "val_ratio": 0.1, "tau_values": [0.1]},
        B=3
    )
    # Models predict log1p value = 0.5 for all samples
    bag.models_by_tau[0.1] = [ConstantModel(0.5), ConstantModel(0.5), ConstantModel(0.5)]
    
    # Simulate member scores (lower error => higher margin)
    member_scores = torch.tensor([0.2, 0.3])  # log1p: ~0.18, ~0.26
    # Simulate non-member scores (higher error => lower margin)
    nonmember_scores = torch.tensor([0.8, 0.9])  # log1p: ~0.59, ~0.64
    
    imgs_member = torch.zeros(2, 3, 32, 32)
    imgs_nonmember = torch.zeros(2, 3, 32, 32)
    
    margin_member, _ = bag.compute_margin(member_scores, imgs_member, tau=0.1)
    margin_nonmember, _ = bag.compute_margin(nonmember_scores, imgs_nonmember, tau=0.1)
    
    # Members should have positive margins (0.5 - log1p(0.2) ≈ 0.32 > 0)
    # Non-members should have negative margins (0.5 - log1p(0.8) ≈ -0.09 < 0)
    assert (margin_member > 0).all(), f"Members should have positive margin, got {margin_member}"
    assert (margin_nonmember < 0).all(), f"Non-members should have negative margin, got {margin_nonmember}"
    
    # Average member margin should be higher than average non-member margin
    assert margin_member.mean() > margin_nonmember.mean()


# DEPRECATED: The old majority vote test is kept for reference but marked as deprecated
def test_bagging_majority_vote_deprecated():
    """DEPRECATED: Test backward compatibility with old voting interface.
    
    Note: This test verifies that the old interface still works,
    but majority voting has been replaced by canonical margin aggregation.
    All evaluation should use the margin m(x) directly, not vote counts.
    """
    bag = BagOfQuantiles(
        base_cfg={"batch_size": 1, "epochs": 1, "lr": 1e-3, "val_ratio": 0.1, "tau_values": [0.1]},
        B=3
    )
    bag.models_by_tau[0.1] = [ConstantModel(0.45), ConstantModel(0.35), ConstantModel(0.65)]
    scores = torch.tensor([0.5, 0.7])
    imgs = torch.zeros(2, 3, 32, 32)
    
    # This should still work but use canonical margin internally
    decisions, diagnostics = bag.decision(scores, imgs, tau=0.1)
    
    # Check backward compatibility fields exist
    assert diagnostics["thresholds_log"].shape == (3, 2)
    assert torch.allclose(torch.expm1(diagnostics["thresholds_log"]), diagnostics["thresholds_raw"], atol=1e-6)
    assert torch.allclose(diagnostics["scores_log"], torch.log1p(scores))
    
    # Margin should be present
    assert "margin" in diagnostics

