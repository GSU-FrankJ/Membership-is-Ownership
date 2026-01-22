import torch

from attacks.scores.t_error import t_error_aggregate, uniform_timesteps


def test_uniform_timesteps_span_range() -> None:
    """Test that uniform timesteps cover the full range [0, T)."""
    steps = uniform_timesteps(T=1000, k=50)
    assert len(steps) == 50
    assert steps[0] == 0
    assert steps[-1] == 999
    assert len(set(steps)) == 50


def test_mean_aggregate(monkeypatch) -> None:
    """Test that t_error_aggregate correctly computes mean across timesteps.
    
    Note: Only mean aggregation is supported. Other aggregation methods
    (q10, q20, weighted) were removed as experiments showed identical results.
    """
    responses = {
        0: torch.tensor([1.0, 2.0]),
        1: torch.tensor([3.0, 4.0]),
        2: torch.tensor([5.0, 6.0]),
    }

    def fake_once(x0, t, model, alphas_bar):
        return responses[t].clone()

    monkeypatch.setattr("attacks.scores.t_error.t_error_once", fake_once)

    x0 = torch.zeros(2, 3, 32, 32)
    timesteps = [0, 1, 2]
    alphas = torch.ones(10)

    mean_scores = t_error_aggregate(x0, timesteps, None, alphas)

    # Expected mean: (1+3+5)/3 = 3.0, (2+4+6)/3 = 4.0
    expected_mean = torch.tensor([3.0, 4.0])
    assert torch.allclose(mean_scores, expected_mean)
