#include<iostream>
#include<chrono>
#include<string>
#include<cstring>
#include "papso_mp.h"
#include "test_functions.h"

template <typename papso_t>
void parallel_async_pso_benchmark(
	 std::size_t iter_per_task
	, optimization_problem_t problem
	, const char* const msg) {
	double avg = 0;
	for (int i = 0; i < 10; ++i) {
		auto t1 = std::chrono::high_resolution_clock::now();
		auto result = papso_t::parallel_async_pso(iter_per_task, problem);
		auto [v, pos] = result.get(); // Could be wasting?
		printf_s("\npar async pso @%s: %lf\n", msg, v);
#ifdef COUNT_STEALING
		//std::printf("steal count: %llu\n", etor.get_steal_count());
#endif
		//auto t2 = std::chrono::high_resolution_clock::now();
		//std::chrono::duration<double> diff = t2 - t1;
		//avg += diff.count();
		//std::cout << "cost time:" << diff.count() << "s" << std::endl;
		//printf("\n");
	}
	//avg /= 10;
	//std::cout << "average time:" << avg << std::endl;
};
