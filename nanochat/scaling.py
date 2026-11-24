"""Helpers for selecting feasible training batches under a FLOP budget."""
import math


def scaling_token_horizon(scaling_params, scaling_ratio, flops, flops_per_token, use_flop_horizon):
    """Use the actual FLOP horizon whenever it determines the training run."""
    if use_flop_horizon:
        if flops <= 0 or flops_per_token <= 0:
            raise ValueError("A positive FLOP horizon is required")
        return flops / flops_per_token
    return int(scaling_ratio * scaling_params)


def cap_batch_for_steps(batch, flops, flops_per_token, min_steps, minimum_batch):
    """Reduce an automatic batch to a sequence-aligned size supporting min_steps.

    A full sequence on every rank is mandatory; raise if this hardware and
    sequence length cannot reach the requested step count at the given budget.
    """
    if min_steps <= 0:
        return batch
    maximum = flops / flops_per_token / min_steps
    if maximum < minimum_batch:
        raise ValueError(f"Cannot reach {min_steps} steps: the budget permits at most {maximum:.0f} tokens per step, below the distributed minimum {minimum_batch}")
    cap = math.floor(maximum / minimum_batch) * minimum_batch
    return min(max(minimum_batch, math.floor(batch / minimum_batch) * minimum_batch), cap)


def fit_local_isoflop(params, losses):
    """Fit exactly two neighbors on either side of the observed minimum."""
    import numpy as np
    x = np.log10(np.asarray(params, dtype=float))
    y = np.asarray(losses, dtype=float)
    if len(x) < 5 or len(set(x)) != len(x) or np.any(np.diff(x) <= 0):
        raise ValueError('need at least five distinct, sorted parameter counts')
    best = int(np.argmin(y))
    if best < 2 or best + 2 >= len(x):
        raise ValueError('observed minimum needs two tested depths on each side; extend the grid')
    fit_x = x[best-2:best+3]
    coeffs = np.polyfit(fit_x, y[best-2:best+3], 2)
    a, b, _ = coeffs
    if a <= 0 or not fit_x[0] < -b/(2*a) < fit_x[-1]:
        raise ValueError('fitted minimum is not bracketed')
    return coeffs, fit_x
