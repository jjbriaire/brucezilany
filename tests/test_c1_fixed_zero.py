"""set_c1_fixed_zero_cf (local NH_Vocoder build nhv5): below the set CF the C1 chirp filter keeps its zeros at rest
(Tan & Carney 2003) instead of re-placing them to hold the CF phase (Zilany & Bruce 2006 Eq. 15), which realises the
cochlear-amplifier gain at low CF. Default 0 = original BEZ2018a."""

import numpy as np

import brucezilany as bz

FS = 100_000


def c1_rms(cf, level, cohc=1.0, dur=0.1):
    t = np.arange(int(dur * FS)) / FS
    x = 20e-6 * 10 ** (level / 20) * np.sqrt(2) * np.sin(2 * np.pi * cf * t)
    stim = bz.stimulus.Stimulus(x, FS, dur * (1 + 1e-9))
    c1 = np.asarray(bz.inner_hair_cell_stages(stim, cf, cohc, 1.0, bz.Species.HUMAN_SHERA))[0]
    seg = c1[len(c1) // 2: int(0.95 * len(c1))]
    return 20 * np.log10(np.sqrt(np.mean(seg ** 2)))


def test_default_is_off():
    assert bz.get_c1_fixed_zero_cf() == 0.0


def test_fixed_zeros_restore_apical_ohc_gain_and_leave_high_cf():
    try:
        moving = c1_rms(125.0, 10.0) - c1_rms(125.0, 10.0, cohc=0.0)
        hi = c1_rms(2000.0, 60.0)
        bz.set_c1_fixed_zero_cf(1000.0)
        fixed = c1_rms(125.0, 10.0) - c1_rms(125.0, 10.0, cohc=0.0)
        assert c1_rms(2000.0, 60.0) == hi  # above the set CF nothing changes
    finally:
        bz.set_c1_fixed_zero_cf(0.0)
    assert moving < 1.0          # original: no amplifier at 125 Hz (cohc 1 -> 0 changes C1 by about -1 dB)
    assert 9.0 < fixed < 14.0    # fixed zeros: about 12 dB


def test_smooth_crossover_has_no_gain_jump():
    """nhv6: raised-cosine blend over one octave above the set CF; the OHC gain (cohc 1 -> 0) has no step at 1 kHz."""
    cfs = [900.0, 990.0, 1010.0, 1100.0, 1400.0, 1900.0, 2100.0]
    try:
        bz.set_c1_fixed_zero_cf(1000.0, 1.0)
        assert bz.c1_zero_weight(999.0) == 0.0 and bz.c1_zero_weight(2001.0) == 1.0
        g = [c1_rms(cf, 10.0) - c1_rms(cf, 10.0, cohc=0.0) for cf in cfs]
        bz.set_c1_fixed_zero_cf(1000.0)  # hard switch (nhv5)
        hard = [c1_rms(cf, 10.0) - c1_rms(cf, 10.0, cohc=0.0) for cf in (990.0, 1010.0)]
    finally:
        bz.set_c1_fixed_zero_cf(0.0)
    assert abs(g[2] - g[1]) < 1.0           # smooth across 1 kHz
    assert hard[0] - hard[1] > 4.0           # the nhv5 hard switch steps there
