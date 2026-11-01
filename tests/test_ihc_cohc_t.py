"""inner_hair_cell_tv / inner_hair_cell_stages_tv: per-sample cohc (local NH_Vocoder build; the MOC hook of
Guest, Farhadi & Carney 2026, Eq. 3: tau_C1 = cohc_t[n] (tau_nl - tau_min) + tau_min)."""

import numpy as np
import pytest

import brucezilany as bz

FS = 100_000


def tone_stim(f=1000.0, level=60.0, dur=0.05):
    t = np.arange(int(dur * FS)) / FS
    x = 20e-6 * 10 ** (level / 20) * np.sqrt(2) * np.sin(2 * np.pi * f * t)
    return bz.stimulus.Stimulus(x, FS, dur * (1 + 1e-9))


@pytest.mark.parametrize("cohc", [1.0, 0.5])
def test_constant_cohc_t_equals_scalar(cohc):
    stim = tone_stim()
    n = stim.n_simulation_timesteps
    a = np.asarray(bz.inner_hair_cell(stim, 1000.0, 1, cohc, 1, bz.Species.HUMAN_SHERA))
    b = np.asarray(bz.inner_hair_cell_tv(stim, 1000.0, np.full(n, cohc), 1, bz.Species.HUMAN_SHERA))
    assert np.array_equal(a, b)
    st = np.asarray(bz.inner_hair_cell_stages_tv(stim, 1000.0, np.full(n, cohc), 1, bz.Species.HUMAN_SHERA))
    assert st.shape == (3, n) and np.array_equal(st[2], a)


def test_cohc_t_step_reduces_c1_after_the_step():
    stim = tone_stim(dur=0.1)
    n = stim.n_simulation_timesteps
    c = np.ones(n)
    c[n // 2:] = 0.3
    full = np.asarray(bz.inner_hair_cell_stages_tv(stim, 1000.0, np.ones(n), 1, bz.Species.HUMAN_SHERA))
    step = np.asarray(bz.inner_hair_cell_stages_tv(stim, 1000.0, c, 1, bz.Species.HUMAN_SHERA))
    rms = lambda a: np.sqrt(np.mean(a ** 2))  # noqa: E731
    assert np.array_equal(full[0, : n // 4], step[0, : n // 4])
    assert rms(step[0, -1000:]) < 0.7 * rms(full[0, -1000:])


def test_cohc_t_short_or_out_of_range_raises():
    stim = tone_stim()
    n = stim.n_simulation_timesteps
    with pytest.raises(ValueError):
        bz.inner_hair_cell_tv(stim, 1000.0, np.ones(n - 1), 1, bz.Species.HUMAN_SHERA)
    bad = np.ones(n)
    bad[10] = 1.5
    with pytest.raises(ValueError):
        bz.inner_hair_cell_tv(stim, 1000.0, bad, 1, bz.Species.HUMAN_SHERA)
    bad[10] = -0.1
    with pytest.raises(ValueError):
        bz.inner_hair_cell_stages_tv(stim, 1000.0, bad, 1, bz.Species.HUMAN_SHERA)
