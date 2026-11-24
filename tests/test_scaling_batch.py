import pytest
from nanochat.scaling import cap_batch_for_steps, scaling_token_horizon, compatible_device_batch


def test_batch_cap_meets_horizon_and_preserves_smaller_batch():
    assert cap_batch_for_steps(524288, 1e15, 1e8, 500, 2048) == 18432
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


@pytest.mark.parametrize("world_size", range(1, 9))
def test_batch_cap_is_sequence_aligned_across_world_sizes(world_size):
    unit = 2048 * world_size
    result = cap_batch_for_steps(524288, 1e16, 1e8, 500, unit)
    assert result % unit == 0
    assert result * 500 <= 1e16 / 1e8
    assert 1e16 / 1e8 / 500 - result < unit


def test_flop_horizon_is_used_only_for_budget_limited_runs():
    assert scaling_token_horizon(3_000_000, 12, 1e15, 1e8, 500) == 10_000_000
    assert scaling_token_horizon(3_000_000, 12, 1e16, 1e8, 500) == 100_000_000
    assert scaling_token_horizon(3_000_000, 12, 1e15, 1e8, 0) == 36_000_000
    with pytest.raises(ValueError):
        scaling_token_horizon(3_000_000, 12, -1, 1e8, 500)


@pytest.mark.parametrize("world_size", range(1, 9))
def test_device_batch_is_largest_compatible_divisor(world_size):
    unit = 2048 * world_size
    for count in (1, 2, 11, 35, 100):
        selected = compatible_device_batch(unit * count, 2048, world_size, 8)
        assert selected == max(i for i in range(1, min(count, 8) + 1) if count % i == 0)
