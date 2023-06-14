/*
* Random number generator with normal distribution within [0, 1)
* Change this to a object?
*/
#ifndef _CANONICAL_RNG
#define _CANONICAL_RNG
#include <random>
#include <memory>
#include <new>
#include <cstdlib>
class canonical_rng
{	
	struct alignas(64) storage {
		
		std::mt19937 generator_;
		std::uniform_real_distribution<double> real_distribute;
	
		storage() :
			generator_(std::random_device{}())
			, real_distribute(0., 1.) {}
	};

	std::unique_ptr<storage> storage_ptr_;
	
public:
	canonical_rng() : storage_ptr_(std::make_unique<storage>()) {}
	canonical_rng(canonical_rng&& oth) noexcept
		: storage_ptr_(std::move(oth.storage_ptr_)) {}
	~canonical_rng() {}

	inline double operator()() const {
		auto& s = *storage_ptr_;
		return s.real_distribute(s.generator_);
	}
};
#endif