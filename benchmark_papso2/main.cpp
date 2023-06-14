#pragma comment ( lib, "Shlwapi.lib" )
#include "../../google_benchmark/include/benchmark/benchmark.h"
#include "../papso2/executor.h"
#include "../papso2/papso2_test.h"


template <size_t Scale> requires (Scale > 0)
struct scaled_rosenbrock {
	static double function(iter beg, iter end) {
		static constexpr auto rosenbrock = test_functions::functions[2];
		volatile double result = 0;
		for (int i = 0; i < Scale; ++i) {
			//benchmark::DoNotOptimize(  );
			result = rosenbrock(beg, end);
		}
		return result;
	}

	static constexpr optimization_problem_t problem{
		&function
		, test_functions::bounds[2]
		, test_functions::dimensions[2]
	};
};

template <size_t Scale>
static void benchmark_scaled_rosenbrock(benchmark::State& state) {
	const optimization_problem_t& problem = scaled_rosenbrock<Scale>::problem;
	const auto min = problem.feasible_bound.first;
	const auto max = problem.feasible_bound.second;
	const auto diff = max - min;

	canonical_rng rng;
	std::vector<double> vec(problem.dimension);
	std::generate(vec.begin(), vec.end(), [&]() {
		return min + rng() * diff;	});

	for (auto _ : state) {
		benchmark::DoNotOptimize(problem.function(vec.cbegin(), vec.cend()));
	}
}
//BENCHMARK_TEMPLATE(benchmark_scaled_rosenbrock, 1);
//BENCHMARK_TEMPLATE(benchmark_scaled_rosenbrock, 550)->Unit(benchmark::kMillisecond);

static void benchmark_executor_create(benchmark::State& state) {
	const auto count = state.range(0);
	for (auto _ : state) {
		benchmark::DoNotOptimize(hungbiu::hb_executor{ static_cast<size_t>(count) });
	}
}
//BENCHMARK(benchmark_executor_create)
//->Unit(benchmark::kMicrosecond)
//->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(6)->Arg(7)->Arg(8);


// Bench speed of optimizing test functions suite
// Args: [fork_count] [iter_per_task] [thread_count] [enable_stealing]
static void benchmark_papso(benchmark::State& state) {
	using papso_t = basic_papso<hungbiu::spmc_buffer<vec_t>, 2, 48, 5000>;

	size_t fork_count = state.range(0);
	size_t iter_per_task = state.range(1);
	hungbiu::hb_executor etor{ 
		static_cast<size_t>(state.range(2))
		, static_cast<bool>(state.range(3)) };

	const optimization_problem_t problem = scaled_rosenbrock<50>::problem;

    for (auto _ : state) {
		auto result = papso_t::parallel_async_pso(etor, fork_count, iter_per_task, problem);
		benchmark::DoNotOptimize(result.get());
    }        
}
// find the best stealing task granularity
//BENCHMARK(benchmark_papso)
//->Iterations(1)
//->Repetitions(10)
//->Unit(benchmark::kMillisecond)
//->Args({ 8, 5000, 8 }) // 6 * 5000
//->Args({ 8, 500, 8 }) //  6 * 500
//->Args({ 16, 200, 8 }) // 3 * 200
//->Args({ 24, 100, 8 }); // 2 * 100

//BENCHMARK(benchmark_papso)
//->Iterations(1)
//->Repetitions(5)
//->Unit(benchmark::kMillisecond)
//->Args({ 1, 5000, 1, 0 }) // base line
//->Args({ 2, 500, 2, 0 }) // NO_WS
//->Args({ 3, 500, 3, 0 })
//->Args({ 4, 500, 4, 0 })
//->Args({ 6, 500, 6, 0 })
//->Args({ 8, 500, 8, 0 })
//->Args({ 2, 500, 2, 1 }) // WS
//->Args({ 3, 500, 3, 1 })
//->Args({ 4, 500, 4, 1 })
//->Args({ 6, 500, 6, 1 })
//->Args({ 8, 500, 8, 1 });

// Args: [function idx] [dimensions] [iterations]
static void benchmark_test_functions(benchmark::State& state) {
	const auto idx = state.range(0);
	const auto dim = test_functions::dimensions[idx];
	const auto min = test_functions::bounds[idx].first;
	const auto max = test_functions::bounds[idx].second;
	const auto diff = max - min;
	canonical_rng rng;

	std::vector<double> vec(dim);
	std::generate(vec.begin(), vec.end(), [&]() {
		return min + rng() * diff;	});

	for (auto _ : state) {
		benchmark::DoNotOptimize(test_functions::functions[idx](vec.cbegin(), vec.cend()));
	}
}
//BENCHMARK(benchmark_test_functions)
//->Arg(0)->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(6)->Arg(7);



template <int N> // N iteartions, Args: [fork_count] [itr_per_task] [dimensions] 
void benchmark_stealing_efficiency(benchmark::State& state) {	
	// A costly object function
	static constexpr auto schwefel_12 = test_functions::functions[1];
	func_t scaled_schwefel_12 = [](iter beg, iter end) {
		for (int i = 0; i < N - 1; ++i) {
			benchmark::DoNotOptimize(schwefel_12(beg, end));
		}
		return schwefel_12(beg, end);
	};

	// Prep args
	const auto dimensions = state.range(2);
	const papso::optimization_problem_t problem{
				scaled_schwefel_12,
				test_functions::bounds[1],
				dimensions
	};
	const auto fork_count = state.range(0);
	const auto itr_per_task = state.range(1);

	// Bench
	hungbiu::hb_executor etor(fork_count);
	for (auto _ : state) {
		auto result = papso::parallel_async_pso(etor, fork_count, itr_per_task, problem);
		benchmark::DoNotOptimize(result.get());
	}
}

//BENCHMARK_TEMPLATE(benchmark_stealing_efficiency, 10)
//->Unit(benchmark::kMillisecond)
//->Args({ 1, 100, 30 })
//->Args({ 2, 100, 30 })
//->Args({ 3, 100, 30 })
//->Args({ 4, 100, 30 })
//->Args({ 5, 100, 30 })
//->Args({ 6, 100, 30 })
//->Args({ 7, 100, 30 })
//->Args({ 8, 100, 30 });

// Args: [thread_count] [fork_count] [itr_per_task] [stealing]
template <size_t NbSize>
static void benchmark_stealing(benchmark::State& state) {	
	using papso_t = basic_papso<hungbiu::spmc_buffer<vec_t>, 2, NbSize, 5000>;

	const auto thread_count = state.range(0);
	const auto fork_count = state.range(1);
	const auto itr_per_task = state.range(2);
	const auto func_index = 1;
	// Bench
	optimization_problem_t problem = scaled_rosenbrock<50>::problem;
	hungbiu::hb_executor etor(thread_count, state.range(3));
	for (auto _ : state) {
		auto result = papso_t::parallel_async_pso(etor, fork_count, itr_per_task, problem);
		benchmark::DoNotOptimize(result.get());
	}
}

//BENCHMARK_TEMPLATE(benchmark_stealing, 50)
//->Unit(benchmark::kMillisecond)
//->Repetitions(10)
//->Args({ 3, 5, 250, 1 }) // WS
//->Args({ 3, 5, 250, 0 }); // NO_WS


//BENCHMARK_TEMPLATE(benchmark_stealing, 75)
//->Unit(benchmark::kMillisecond)
//->Repetitions(1)
//// WS
//->Args({ 4, 15, 250, 1 })
//// NO_WS
//->Args({ 4, 15, 250, 0 });


BENCHMARK_TEMPLATE(benchmark_stealing, 50)
->Unit(benchmark::kMillisecond)
->Repetitions(5)
// WS
->Args({ 7, 10, 500, 1 })
// NO_WS
->Args({ 7, 10, 500, 0 });

BENCHMARK_TEMPLATE(benchmark_stealing, 80)
->Unit(benchmark::kMillisecond)
->Repetitions(5)
// WS
->Args({ 7, 16, 500, 1 })
// NO_WS
->Args({ 7, 16, 500, 0 });

BENCHMARK_TEMPLATE(benchmark_stealing, 100)
->Unit(benchmark::kMillisecond)
->Repetitions(5)
// WS
->Args({ 7, 20, 500, 1 }) 
// NO_WS
->Args({ 7, 20, 500, 0 });

// Args: [fork_count]
template <int sz> requires (sz > 0)
double time_scaled_func(iter beg, iter end) { // 420ns
	for (int i = 0; i < sz - 1; ++i) {
		test_functions::functions[3](beg, end);
	}
	return test_functions::functions[3](beg, end);
}
template <size_t nb_sz>
void benchmark_particle_communication(benchmark::State& state) {	
	using papso_t = basic_papso<hungbiu::spmc_buffer<vec_t>, nb_sz, 48, 5000>;
	
	optimization_problem_t problem{
		test_functions::functions[3],
		test_functions::bounds[3],
		state.range(2)
	};

	size_t fork_count = static_cast<size_t>(state.range(0));
	size_t iter_per_task = static_cast<size_t>(state.range(1));

	hungbiu::hb_executor etor(fork_count);
	for (auto _ : state) {
		auto result = papso_t::parallel_async_pso(etor, fork_count, iter_per_task, problem);
		benchmark::DoNotOptimize(result.get());
	}
}

// Neighbor hood size
//BENCHMARK_TEMPLATE(benchmark_particle_communication, 2u) // lbest
//->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(10)
//->Args({ 16, 500, 30 })
//->Args({ 16, 500, 50 })
//->Args({ 16, 500, 70 })
//->Args({ 16, 500, 100 });
//BENCHMARK_TEMPLATE(benchmark_particle_communication, 8u)
//->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(10)
//->Args({ 16, 500, 30 })
//->Args({ 16, 500, 50 })
//->Args({ 16, 500, 70 })
//->Args({ 16, 500, 100 });
//BENCHMARK_TEMPLATE(benchmark_particle_communication, 16u)
//->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(10)
//->Args({ 16, 500, 30 })
//->Args({ 16, 500, 50 })
//->Args({ 16, 500, 70 })
//->Args({ 16, 500, 100 });
//BENCHMARK_TEMPLATE(benchmark_particle_communication, 24u)
//->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(10)
//->Args({ 16, 500, 30 })
//->Args({ 16, 500, 50 })
//->Args({ 16, 500, 70 })
//->Args({ 16, 500, 100 });
//
//BENCHMARK_TEMPLATE(benchmark_particle_communication, 32u)
//->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(10)
//->Args({ 16, 500, 30 })
//->Args({ 16, 500, 50 })
//->Args({ 16, 500, 70 })
//->Args({ 16, 500, 100 });
//
//BENCHMARK_TEMPLATE(benchmark_particle_communication, 40u) // gbest
//->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(10)
//->Args({ 16, 500, 30 })
//->Args({ 16, 500, 50 })
//->Args({ 16, 500, 70 })
//->Args({ 16, 500, 100 });

// Communication cost - Forks
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 40u, 1)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Repetitions(10)
//->Arg(2)->Arg(4)->Arg(6)->Arg(8);
//
//
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 40u, 1)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Repetitions(10)
//->Arg(2)->Arg(4)->Arg(6)->Arg(8);

// Commnucation cost - Swarm size
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 40u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 50u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 60u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 70u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, naive_buffer, 80u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 40u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 50u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 60u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 70u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);
//BENCHMARK_TEMPLATE(benchmark_particle_communication, my_buffer, 80u)
//->Unit(benchmark::kMillisecond)->Iterations(10)->Arg(8);




BENCHMARK_MAIN();