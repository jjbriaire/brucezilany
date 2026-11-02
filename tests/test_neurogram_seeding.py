"""Neurogram seeding (nhv4): every (CF, fibre, trial) draws its own random stream.

Before: utils::SEED + (1 + cf) * (1 + trial) * 1500 for every fibre of a CF, so the fibres of one CF and trial drew the
same random stream (correlated spike trains: the streams diverge only through each fibre's own spont and refractory
parameters) and (cf, trial) pairs with equal products collided.
"""

import numpy as np
import pytest

import brucezilany as bz


def _stim():
    fs = 100_000
    t = np.arange(int(0.05 * fs)) / fs
    x = np.sin(2 * np.pi * 1000 * t) * 20e-6 * 10 ** (60 / 20)
    return bz.stimulus.Stimulus(x, fs, 0.05)


def test_fiber_seed_unique_and_legacy():
    bz.set_seed(42)
    bz.set_legacy_seeding(False)
    seeds = {bz.fiber_seed(c, f, t) for c in range(4) for f in range(6) for t in range(3)}
    assert len(seeds) == 4 * 6 * 3
    bz.set_legacy_seeding(True)
    try:
        assert bz.fiber_seed(1, 0, 3) == bz.fiber_seed(3, 0, 1) == 42 + 2 * 4 * 1500  # the old collision
        assert bz.fiber_seed(0, 0, 0) == bz.fiber_seed(0, 5, 0)  # the old shared stream across fibres
    finally:
        bz.set_legacy_seeding(False)


def test_legacy_and_fixed_differ():
    """The two seedings give different (both reproducible) neurograms; the legacy one is the pre-nhv4 output."""
    stim = _stim()
    outs = {}
    for legacy in (True, False):
        for _ in range(2):
            bz.set_seed(7)
            bz.set_legacy_seeding(legacy)
            try:
                ng = bz.Neurogram([1000.0, 2000.0], n_low=1, n_med=1, n_high=4, n_threads=2)
                ng.bin_width = stim.time_resolution
                ng.create(stim, n_rep=1, n_trials=2)
                outs.setdefault(legacy, []).append(np.asarray(ng.get_output()))
            finally:
                bz.set_legacy_seeding(False)
    assert np.array_equal(*outs[True]) and np.array_equal(*outs[False])
    assert not np.array_equal(outs[True][0], outs[False][0])


def test_reproducible():
    stim = _stim()
    outs = []
    for _ in range(2):
        bz.set_seed(11)
        ng = bz.Neurogram([500.0, 2000.0], n_low=1, n_med=1, n_high=2, n_threads=2)
        ng.bin_width = stim.time_resolution
        ng.create(stim, n_rep=1, n_trials=2)
        outs.append(np.asarray(ng.get_output()))
    assert np.array_equal(outs[0], outs[1])
    assert not np.array_equal(outs[0][:, 0], outs[0][:, 1])  # trials differ
