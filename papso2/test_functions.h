#ifndef _TEST_FUNCTIONS
#define _TEST_FUNCTIONS

#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <tuple>
#include <array>
#define PRINT
#undef PRINT
#ifdef PRINT
#include <cstdio>
#endif
struct test_functions {
	// 16 digits Pi
	static constexpr double Pi = 245850922.0 / 78256779.0;

	using iter = std::vector<double>::const_iterator;
	using test_function_type = double(*)(iter, iter);

	// Class 1

	// f1
	static double sphere(iter beg, iter end) {
		double sum = 0;
		while (beg != end) {
			sum += std::pow(*beg, 2);
			beg++;
		}
		return sum;
	}

	// f2 
	static double schwefel_12(iter beg, iter end) {
		double squares_sum = 0;
		double partial_sum = 0;

		// next : [beg + 1, end)
		while (beg != end) {
			partial_sum += *beg;			
			squares_sum += std::pow(partial_sum, 2);
			++beg;
		}
		return squares_sum;
	}

	// f3
	static double rosenbrock(iter beg, iter end) {
		double sum = 0;
		iter next = beg + 1;
		while (next != end) {
			sum += 100 * std::pow(*next - std::pow(*beg, 2), 2)
				+ std::pow(*beg, 2);
			beg++;
			next++;
		}
		return sum;
	}

	// Class 2

	// f4
	static double schwefel_26(iter beg, iter end) {
		double sum = 0;
		double dim = end - beg;
		while (beg != end) {
			sum += (*beg) * std::sin(std::sqrt(std::abs(*beg)));
			beg++;
		}
		return sum / dim;
	}

	// f5
	static double rastrigin(iter beg, iter end) {
		double sum = 0;
		while (beg != end) {
			sum += std::pow(*beg, 2)
				- 10 * std::cos(2 * Pi * (*beg))
				+ 10;

			beg++;
		}
		return sum;
	}

	// f6
	static double ackley(iter beg, iter end)
	{
		double square_sum = 0;
		double cos_sum = 0;
		auto dim = end - beg;
		while (beg != end) {
			square_sum += std::pow(*beg, 2);
			cos_sum += std::cos(2 * Pi * (*beg));
			beg++;
		}
		return -20 * std::exp(-0.2 * std::sqrt(square_sum / dim))
			- std::exp(cos_sum / dim)
			+ 22.718282;
	}

	// f7
	static double griewank(iter beg, iter end) {

		double sum = 0.;
		double product = 1.;
		iter it = beg;
		while (it != end) {
			sum += pow(*it, 2);
			product *= cos(*it / sqrt((it - beg) + 1));
			++it;
		}

		return sum / 4000. - product + 1.;
	}

	// f8

	// f9

	static constexpr std::array functions = {
		sphere, schwefel_12, rosenbrock, schwefel_26, rastrigin,
		ackley, griewank
	};
	static constexpr unsigned dimensions[] = {
		30u, 30u, 30u, 30u, 30u, 30u, 30u
	};
	static constexpr std::pair<double, double> bounds[] = {
		{-100., 100.}, {-100., 100.}, {-30., 30.}, {-500., 500.},
		{-5.12, 5.12}, {-32., 32.}, {-600., 600.}
	};
	static constexpr const char* function_names[] = {
		"sphere", "schwefel 1.2", "rosenbrock", "schwefel 2.6",
		"rastrigin", "ackley", "griewank"
	};
};

#endif