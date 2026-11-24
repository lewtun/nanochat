import pytest
from nanochat.scaling import cap_batch_for_steps


def test_batch_cap_meets_horizon_and_preserves_smaller_batch():
    assert cap_batch_for_steps(524288, 1e15, 1e8, 500, 2048) == 16384
    assert cap_batch_for_steps(8192, 1e15, 1e8, 500, 2048) == 8192


def test_impossible_budget_is_rejected():
    with pytest.raises(ValueError, match='Cannot reach'):
        cap_batch_for_steps(524288, 1e15, 2e9, 500, 2048)


def test_local_fit_ignores_distant_points_and_rejects_edges():
    import numpy as np
    from nanochat.scaling import fit_local_isoflop
    x = np.arange(7, dtype=float)
    loss = (x - 3.0) ** 2
    loss[0] = 1000
    loss[-1] = 2000
    coefficients, window = fit_local_isoflop(10 ** x, loss)
    assert np.allclose(coefficients, [1, -6, 9])
    assert list(window) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match='two tested depths'):
        fit_local_isoflop(10 ** x, x)
