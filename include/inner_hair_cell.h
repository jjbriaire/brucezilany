#pragma once


#include <array>
#include <complex>
#include <vector>

#include "types.h"
#include "stimulus.h"

namespace ihc
{
	/**
	* NH_Vocoder (nhv5/nhv6): up to C1_FIXED_ZERO_CF (Hz) the C1 chirp filter keeps its zeros at their rest position instead of
	* re-placing them every sample to hold the phase at CF (Zilany & Bruce 2006 Eq. 15). Re-placing the zeros cancels the pole-shift
	* gain at low CF (no apical amplifier); fixed zeros with an equal pole shift is the Tan & Carney (2003) design (level-independent
	* glide). Between C1_FIXED_ZERO_CF and C1_FIXED_ZERO_CF * 2^C1_FIXED_ZERO_OCT the zero position is a raised-cosine (in log CF)
	* blend of rest and re-placed (nhv6; octaves 0 = the nhv5 hard switch). C1_FIXED_ZERO_CF 0 (default) = original BEZ2018a.
	*/
	extern double C1_FIXED_ZERO_CF;
	extern double C1_FIXED_ZERO_OCT;
	void set_c1_fixed_zero_cf(double cf_hz, double octaves = 0.0);
	double get_c1_fixed_zero_cf();
	double get_c1_fixed_zero_octaves();
	//! weight of the re-placed zero position at this CF: 0 = fixed (rest), 1 = re-placed (original)
	double c1_zero_weight(double cf_hz);

	/* nhv7: lower clip of the cochlear-amplifier gain law (dB; original BEZ2018a 15). The law
	 * clip(26 (tanh(2.2 log10(CF / 600) + 0.15) + 1), floor, 60) sets the range of the C1 and control-path time constants, i.e. the
	 * OHC gain (cohc 1 -> 0) and the compression at that CF; the low-level (tau_max) tuning is unchanged. */
	extern double GAIN_FLOOR_DB;
	void set_gain_floor_db(double db);
	double get_gain_floor_db();
	double ohc_gain_db(double cf);

	double delay_cat(double cf);

	struct MiddleEarFilter
	{
		double gain_max;
		std::array<std::array<double, 6>, 3> m;
		std::array<double, 3> mey1, mey2, mey3, px_cache;
		size_t i;

		MiddleEarFilter(double time_resolution, Species species);

		double operator()(double px);
	};

	struct WideBandGammaToneFilter
	{
		double time_resolution;
		double center_freq;
		int order;
		double cf;
		double delta_phase;
		double phase;
		std::array<std::complex<double>, 4> gtf;
		std::array<std::complex<double>, 4> gtf_last;

		double tau_wb_max;
		double tau_wb_min;
		double tau_wb;
		double bm_tau_min;
		double bm_tau_max;
		double tau_max;
		double tau_min;
		double ratio_bm;
		double wb_gain;

		double tmpcos;

		std::vector<double> gain_delay;

		WideBandGammaToneFilter(
			double time_resolution,
			double cf,
			Species species,
			double cohc
		);

		double operator()(double me_out);

		//! shift of the location of poles of the C1 filter from the initial positions
		void shift_poles(double tau_c1, size_t n);

		[[nodiscard]] std::pair<double, int> calculate_gain_delay() const;
	};

	struct BoltzmanFilter
	{
		double asymptote;

		double operator()(double wb_out) const;
	};

	template <int Order>
	struct LowPassFilter
	{
		double c1_lp;
		double c2_lp;
		std::array<double, Order + 1> ohc;
		std::array<double, Order + 1> ohc_last;

		LowPassFilter(double time_resolution, double fc);

		double operator()(double x);
	};

	struct PostOhcFilter
	{
		double cohc;
		double bm_tau_min;
		double bm_tau_max;
		double min_r;
		double s0;

		PostOhcFilter(double cohc, double bm_tau_min, double bm_tau_max, double asym);

		double operator()(double x) const;
		//! local NH_Vocoder build: the same with the OHC factor of this sample (time-varying cohc, MOC hook)
		double operator()(double x, double cohc_now) const;
	};

	struct LogarithmicTransductionFunction
	{
		double slope;
		double asymptote;
		double strength;

		LogarithmicTransductionFunction(double slope, double asym);

		double operator()(double x) const;
	};

	struct ChirpFilter
	{
		double time_resolution;
		double cf;
		double tau_max;

		double init_phase;
		double gain_norm;

		std::array<std::array<double, 4>, 12> input;
		std::array<std::array<double, 4>, 12> output;
		std::array<std::complex<double>, 11> phase_array;

		double sigma0;
		double ipw;
		double ipb;
		double rpa;
		double r0;
		double fs_bi_linear;
		double gain_denominator;
		bool c1;
		double zero_w; //!< 1 = zeros re-placed every sample (original); 0 = fixed at r0_rest; between: blend
		double r0_rest;

		ChirpFilter(double time_resolution, double cf_in, double bm_tau_max, bool c1 = true);

		void fill_phase_array(double r_sigma, bool multiply = false);

		double operator()(double me_out, double r_sigma);
	};
}


/**
 * @brief model_IHC_BEZ2018 - Bruce, Erfani & Zilany (2018) Auditory Nerve Model
 *
 *     vihc = model_IHC_BEZ2018(pin,CF,n_rep,dt,rep_time,cohc,cihc,species);
 *
 * vihc is the inner hair cell (IHC) relative transmembrane potential (in volts)

 * For example,
 *    vihc = model_IHC_BEZ2018(pin,1e3,10,1/100e3,0.2,1,1,2); **requires 8 input arguments
 *
 * models a normal human fiber of high spontaneous rate (normal OHC & IHC function) with a CF of 1 kHz,
 * for 10 repetitions and a sampling rate of 100 kHz, for a repetition duration of 200 ms, and
 * with approximate implementation of the power-law functions in the synapse model.
 *
 *
 * NOTE ON SAMPLING RATE:-
 * Since version 2 of the code, it is possible to create the model at a range
 * of sampling rates between 100 kHz and 500 kHz.
 * It is recommended to create the model at 100 kHz for CFs up to 20 kHz, and
 * at 200 kHz for CFs> 20 kHz to 40 kHz.
 *
 * @param stimulus the input sound wave
 * @param cf characteristic frequency
 * @param n_rep the number of repetitions for the psth
 * @param cohc is the OHC scaling factor: 1 is normal OHC function; 0 is complete OHC dysfunction
 * @param cihc is the IHC scaling factor: 1 is normal IHC function; 0 is complete IHC dysfunction
 * @param species is the model species: "1" for cat, "2" for human with BM tuning from Shera et al. (PNAS 2002),
 *    or "3" for human BM tuning from Glasberg & Moore (Hear. Res. 1990)
 * @returns 
 */
std::vector<double> inner_hair_cell(
	const stimulus::Stimulus& stimulus,
	double cf = 1e3,
	int n_rep = 10,
	double cohc = 1,
	double cihc = 1,
	Species species = HUMAN_SHERA
);

/**
 * Local NH_Vocoder build: the C1 and C2 chirp-filter outputs (basilar-membrane signals before their
 * transduction functions) next to the IHC output, for one repetition.
 *
 * @returns 3 x n_simulation_timesteps values, row-major: C1 output, C2 output, IHC output (identical to
 *    inner_hair_cell with n_rep = 1); every row shifted by the same delay_point.
 */
std::vector<double> inner_hair_cell_stages(
	const stimulus::Stimulus& stimulus,
	double cf = 1e3,
	double cohc = 1,
	double cihc = 1,
	Species species = HUMAN_SHERA
);

/**
 * Local NH_Vocoder build: inner_hair_cell (n_rep = 1) with a per-sample OHC factor cohc_t (the MOC hook of
 * Guest, Farhadi & Carney 2026, Eq. 3: tau_C1[n] = cohc_t[n] (tau_nl[n] - tau_min) + tau_min, i.e. cohc_t = c(t) COHC).
 * The wide-band control-path filter is initialised with cohc_t[0]. cohc_t needs at least
 * n_simulation_timesteps values in [0, 1] (std::invalid_argument otherwise). A constant cohc_t reproduces the
 * scalar call exactly.
 */
std::vector<double> inner_hair_cell_tv(
	const stimulus::Stimulus& stimulus,
	double cf,
	const std::vector<double>& cohc_t,
	double cihc = 1,
	Species species = HUMAN_SHERA
);

//! Local NH_Vocoder build: inner_hair_cell_stages with a per-sample OHC factor (see inner_hair_cell_tv).
std::vector<double> inner_hair_cell_stages_tv(
	const stimulus::Stimulus& stimulus,
	double cf,
	const std::vector<double>& cohc_t,
	double cihc = 1,
	Species species = HUMAN_SHERA
);
