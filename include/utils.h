#pragma once
#include <complex>
#include <random>
#include <valarray>
#include <vector>
#include <iostream>
#if defined(_MSC_VER) 
#include <corecrt_math_defines.h>
#endif
#include <cmath>
#include "types.h"

#ifndef PROJECT_ROOT
#define PROJECT_ROOT "project_root_not_defined"
#endif

namespace utils
{
	/**
	 * Helper to validate the value of a parameter at runtime
	 * @tparam T Type of the parameter
	 * @param p the value of the parameter
	 * @param lb the lower bound
	 * @param ub the upper bound
	 * @param name the name of the parameter
	 */
	template <typename T>
	void validate_parameter(const T p, const T lb, const T ub, const std::string& name = "")
	{
		if (p < lb || p > ub)
		{
			std::ostringstream ss;
			ss << name << "= " << p << " is out of bounds [" << lb << ", " << ub << "]";
			std::cout << ss.str() << std::endl;
			throw std::invalid_argument(ss.str());
		}
	}

	/**
	 * The fast-fourier transform of a signal x
	 * @param x signal to transform
	 */
	void fft(std::valarray<std::complex<double>>& x);

	/**
	 * The inverse fast-fourier transform of a signal x
	 * @param x signal to transform
	 */
	void ifft(std::valarray<std::complex<double>>& x);

	//! The random seed
	extern int SEED;

	/**
	* Set the global seed
	* @param seed the new value of the random seed
	*/
	void set_seed(int seed);

	//! Pre-nhv4 neurogram seeding (SEED + (1 + cf) * (1 + trial) * 1500 for every fibre of a CF): for comparison only
	extern bool LEGACY_SEEDING;

	/**
	* Use the pre-nhv4 neurogram seeding, in which all fibres of one CF and trial drew the same random stream and
	* (cf, trial) pairs with equal products collided. Off by default; only to reproduce old neurograms.
	*/
	void set_legacy_seeding(bool legacy);

	/**
	* The seed of one fibre's spike train in a neurogram: SEED mixed with the CF index, the fibre index within the CF
	* and the trial index (splitmix64 finaliser), so every (cf, fibre, trial) draws its own stream.
	*/
	size_t fiber_seed(size_t cf_i, size_t fiber_i, size_t trial_i);

	struct RandomGenerator {
		std::mt19937 gen;
		std::uniform_real_distribution<double> uniform;
		std::normal_distribution<double> normal;

		RandomGenerator(const size_t seed): gen(seed), uniform(0, 1.0), normal(0, 1.0) {}

		/**
		* Generate a single uniform random number in [0, 1)
		* @return the number
		*/
		double rand1() 
		{
			return uniform(gen);
		}

		/**
		* Generate a single standard normal (mu = 0, sigma = 1) random number
		* @return the number
		*/
		double randn1() 
		{
			return normal(gen);
		}

		//! Fill a gaussian vector
		void fill_gaussian(std::vector<double> &x) 
		{
			for (auto &xi : x)
				xi = randn1();
		}
	};

	/**
	* @brief Fast (exact) fractional Gaussian noise and Brownian motion generator for a fixed Hurst
	* index of .9 and a fixed time resolution (tdres) of 1e-4.
	*
	*
	* @param n_out is the length of the output sequence.
	* @param noise type of random noise
	* @param mu the mean of the noise
	* @return returns a sequence of fractional Gaussian noise with a standard deviation of one.
	*/
	std::vector<double> fast_fractional_gaussian_noise(
		RandomGenerator& rng,
		int n_out = 5300,
		NoiseType noise = RANDOM,
		double mu = 100
	);

	/**
	 * Bin a vector in n_bins, i.e. digitize a vector and sum over the data in a single bin
	 * @param x the input signal
	 * @param n_bins  the number of bins
	 * @return the binned vector
	 */
	std::vector<double> make_bins(const std::vector<double>& x, size_t n_bins);

	/**
	 * Cumulative sum of a vector
	 * @param x the vector
	 * @return the cumulative summed vector
	 */
	std::vector<double> cum_sum(const std::vector<double>& x);

	/**
	 * Add element-wise the contents of y to x
	 * @param x the storage vector
	 * @param y the vector to be added
	 */
	void add(std::vector<double>& x, const std::vector<double>& y);

	/**
	 * Scale the elements of a vector x by a constant y
	 * @param x the vector
	 * @param y the scalar
	 */
	void scale(std::vector<double>& x, double y);

	/**
	 * Compute the sum of a vector x
	 * @param x the vector
	 * @return the mean
	 */
	double sum(const std::vector<double>& x);

	/**
	 * Compute the (sample) variance of a vector x
	 * @param x the vector
	 * @param m the mean of the vector
	 * @return the variance
	 */
	double variance(const std::vector<double>& x, double m);


	/**
	 * Compute the (sample) std. deviation of a vector x
	 * @param x the vector
	 * @param m the mean of the vector
	 * @return the std. deviation
	 */
	double std(const std::vector<double>& x, double m);


	/**
	 * Compute the (sample) mean of a vector x
	 * @param x the vector
	 * @return the mean 
	 */
	double mean(const std::vector<double>& x);

	/**
	 * Reduce a 2d vector to a 1d vector of means
	 * @param x the 2d vector
	 * @return a 1d vector containing means
	 */
	std::vector<double> reduce_mean(const std::vector<std::vector<double>>& x);

	/**
	 * Reduce a 2d vector to a 1d vector of std. deviations
	 * @param x the 2d vector
	 * @return a 1d vector containing std. deviations
	 */
	std::vector<double> reduce_std(const std::vector<std::vector<double>>& x, const std::vector<double>& means);

	/**
	 * print a vector of type T to std::cout. 
	 * @tparam T type of the vector elements, should be compatible with std::ostream
	 * @param x the vector
	 */
	template <typename T>
	void print(const std::vector<T>& x)
	{
		for (auto& xi : x)
		{
			std::cout << xi << " ";
		}
		std::cout << std::endl;
	}

	/**
	 * Generate a log_space, works similar to how np.log_space or matlabs log_space works.
	 *
	 * @param start start of the range
	 * @param end end of the range
	 * @param n the number of points to be generated
	 * @return a vector of n log10 space points
	 */
	std::vector<double> log_space(double start, double end, size_t n);

	/**
	 * Generate a hamming window of n coefficients
	 * @param n the number of coefficients
	 * @return the hamming window
	 */
	std::vector<double> hamming(size_t n);

	/**
	 * Apply a 1D FIR digital filter to a signal using a filter with coefficients
	 * @param coefficients the coefficients to use in the filter
	 * @param signal the signal to filter
	 * @return the filtered signal.
	 */
	std::vector<double> filter(const std::vector<double>& coefficients, const std::vector<double>& signal);

	/**
	 * Plotting function (requires local python setup)
	 * @param vectors data to plot
	 * @param ptype type of plot
	 * @param title the title
	 * @param xlabel the xlabel
	 * @param ylabel the ylabel
	 * @param detach whether to detach from the python proc, if false, execution is paused. 
	 */
	void plot(
		std::vector<std::vector<double>> vectors,
		const std::string& ptype = "line",
		const std::string& title = "title",
		const std::string& xlabel = "x",
		const std::string& ylabel = "y",
		const std::string& extra = "",
		bool detach = true
	);

	/**
	 * Get a subsequence of a vector
	 * @tparam T the type of the vector
	 * @param input the input vector
	 * @param start the starting position
	 * @param stop the ending [position
	 * @param step the step interval
	 * @return 
	 */
	template<typename T>
	std::vector<T> subsequence(const std::vector<T>& input, const size_t start, const size_t stop, const size_t step)
	{
		std::vector<T> result;

		if (!input.empty() && step > 0) 
			result.reserve((stop - start + step - 1) / step);

		for (size_t i = start + step - 1; i < stop; i += step) 
			result.push_back(input[i]);
		
		return result;
	}
}
