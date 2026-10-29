"""inner_hair_cell_stages: C1 / C2 chirp-filter outputs next to the IHC output (local NH_Vocoder build)."""

import numpy as np
import pytest

import brucezilany as bz

FS = 100_000


def tone_stim(f=1000.0, level=60.0, dur=0.05):
    t = np.arange(int(dur * FS)) / FS
    x = 20e-6 * 10 ** (level / 20) * np.sqrt(2) * np.sin(2 * np.pi * f * t)
    return bz.stimulus.Stimulus(x, FS, dur * (1 + 1e-9))


@pytest.mark.parametrize("cf", [1000.0, 100.0])
def test_stages_row2_equals_inner_hair_cell(cf):
    stim = tone_stim()
    st = np.asarray(bz.inner_hair_cell_stages(stim, cf, 1, 1, bz.Species.HUMAN_SHERA))
    ihc = np.asarray(bz.inner_hair_cell(stim, cf, 1, 1, 1, bz.Species.HUMAN_SHERA))
    assert st.shape == (3, ihc.size)
    assert np.array_equal(st[2], ihc)


def test_c1_peaks_at_cf():
    stim = tone_stim()
    rms = lambda cf: np.sqrt(np.mean(np.asarray(bz.inner_hair_cell_stages(stim, cf))[0] ** 2))  # noqa: E731
    assert rms(1000.0) > 3 * rms(4000.0)
