#ifndef _PAPSO
#define _PAPSO

#include <vector>
#include <tuple>
#include <variant>
#include <atomic>
#include <memory>
#include <numeric>
#include <type_traits>
#include <mutex>
#include <condition_variable>
#include <future>
#include<omp.h>
//#include "executor.h"
#include "spmc_buffer.h"
#include "canonical_rng.h"

using vec_t = std::vector<double>;
using iter = vec_t::const_iterator;
using func_t = double(*)(iter, iter);
using bound_t = std::pair<double, double>;

struct optimization_problem_t {
	const func_t function;
	bound_t feasible_bound;
	size_t dimension;
};

template <typename buffer_t, size_t neighbor_size, size_t swarm_size, size_t iteration>
class basic_papso {
	class alignas(64) aligned_atomic_double {
		std::atomic<double> value_;
	public:
		aligned_atomic_double(double desired = std::numeric_limits<double>::max())
			: value_(desired) {}
		aligned_atomic_double(aligned_atomic_double&& oth) noexcept
			: value_(oth.load()) {}
		~aligned_atomic_double() {}

		double load() const noexcept {
			return value_.load(std::memory_order_acquire);
		}
		void store(double desired) noexcept {
			value_.store(desired, std::memory_order_release);
		}
	};

	// particle
	struct my_particle {
		double value;
		double best_value;
		vec_t velocity;
		vec_t position;
		vec_t best_position;
	};
public:

	using particle = my_particle;
	using atomic_double = aligned_atomic_double;
	using size_t = std::size_t;
	using range_t = std::pair<size_t, size_t>;
	//using worker_handle = hungbiu::hb_executor::worker_handle;

	static constexpr size_t swarm_size = swarm_size;

private:

	const func_t f;
	size_t dimension;
	double min, max;
	size_t iteration_per_task;
	std::atomic<particle*> gbest = { nullptr };
	std::vector<particle> particles;

	//--------------------------------
	// Synchronization
	std::vector<atomic_double> best_values;
	std::vector<buffer_t> best_positions;
	std::vector<canonical_rng> rngs;
	//--------------------------------

	std::mutex completion_mtx;
	std::condition_variable completion_cv;
	size_t forks = 0;

	class fork_tracer {
		basic_papso* state_ptr;
	public:
		fork_tracer(basic_papso* p)
			: state_ptr(p) {
			std::lock_guard guard{ p->completion_mtx };
			p->forks++;
		}
		fork_tracer(fork_tracer&& oth) noexcept
			: state_ptr(std::exchange(oth.state_ptr, nullptr)) {}
		~fork_tracer() {
			if (!state_ptr) {
				return;
			}

			bool is_completed = false;
			{ // Critical section
				std::lock_guard guard{ state_ptr->completion_mtx };
				state_ptr->forks--;
				if (0 == state_ptr->forks) {
					is_completed = true;
				}
			}

			if (is_completed) {
				state_ptr->completion_cv.notify_one();
			}
		}
	};

public:
	basic_papso(const func_t f, const bound_t& bounds, size_t dim, size_t iter_per_task) :
		f(f),
		dimension(dim), min(bounds.first), max(bounds.second),
		iteration_per_task(iter_per_task) {}
	basic_papso(const basic_papso&) = delete;

private:
	void initialize_state(size_t fork_count) { // Tons of allocations
		particles.resize(swarm_size);
		best_values.resize(swarm_size);
		best_positions.resize(swarm_size);
		rngs.resize(fork_count);

		for (size_t i = 0; i < swarm_size; ++i) {
			particle& p = particles[i];
			p.position.resize(dimension);
			p.best_position.resize(dimension);
			p.velocity.resize(dimension);
			evaluate_particle(i);
		}
	}

	void evaluate_particle(size_t i) noexcept {
		// Evaluate
		particle& p = particles[i];
		p.value = f(p.position.cbegin(), p.position.cend());

		// Update pbest
		if (p.value < p.best_value) {
			p.best_value = p.value;
			p.best_position = p.position;

			// Publish
			best_values[i].store(p.value);
			best_positions[i].put(p.best_position);
		}
	}

	void initialize_swarm(canonical_rng& rng) { // Must evaluate particles first!
		auto random_xi = [&]() {
			return min + rng() * (max - min);
		};

		for (size_t i = 0; i < swarm_size; ++i) { // particle i
			particle& p = particles[i];
			for (size_t j = 0; j < dimension; ++j) { // dimension j
				p.position[j] = random_xi();
				p.best_position[j] = p.position[j];
				p.velocity[j] = (random_xi() - p.position[j]) / 2.0;
			}

			p.best_value = p.value = f(p.position.cbegin(), p.position.cend());

			// Publish
			best_values[i].store(p.best_value);
			best_positions[i].put(p.best_position);
		}
	}

	particle& update_gbest() noexcept { // Thread safe!
		particle* best_ptr = &particles.front();
		double best_val = best_values.front().load();

		for (size_t i = 0; i < swarm_size; ++i) {
			double v = best_values[i].load();
			if (v < best_val) {
				best_ptr = &particles[i];
				best_val = v;
			}
		}
		gbest.store(best_ptr, std::memory_order_release);
		return *best_ptr;
	}

	const vec_t& get_lbest_unsafe(int idx) const noexcept {
		const particle* lbest_ptr = &particles[idx]; // !!Middle of neighbor
		const int max_offset = neighbor_size / 2; // Always positive
		// offset: [-max_offset, +max_offset]
//#pragma omp parallel for
		for (int offset = -max_offset; offset <= max_offset; ++offset) {
			size_t neighbor = (idx + swarm_size + offset) % swarm_size; // in case (idx + offset) < 0
			if (particles[neighbor].best_value < lbest_ptr->best_value) {
				lbest_ptr = &particles[neighbor];
			}
		}
		return lbest_ptr->best_position;
	}

	using var_t = std::variant<const vec_t*, typename buffer_t::viewer>;
	var_t get_lbest(int idx, const range_t range) noexcept { // Thread safe!
		size_t lbest_idx = idx;	// !!Middle of neighbor
		double lbest_val = particles[idx].best_value;
		const int max_offset = neighbor_size / 2; // Always positive

		auto in_range = [&](size_t i) -> bool {
			return range.first <= i && i < range.second;
		};

		// Traverse neighborhood
		// offset: [-max_offset, +max_offset]
		for (int offset = -max_offset; offset <= max_offset; ++offset) {
			size_t neighbor = (idx + swarm_size + offset) % swarm_size; // in case (idx + offset) < 0

			double v = in_range(neighbor)
				? particles[neighbor].best_value
				: best_values[neighbor].load();

			if (v < lbest_val) {
				lbest_val = v;
				lbest_idx = neighbor;
			}
		}

		// Return
		if (in_range(lbest_idx)) {
			return &particles[lbest_idx].best_position;
		}
		else {
			return best_positions[lbest_idx].get();
		}
	}

	void move_particle(size_t idx, var_t lbest_var, canonical_rng* rng_ptr) {
		auto calculate_velocity = [rng_ptr](double vi, double xi, double pbest, double lbest) {
			static constexpr double INERTIA = 0.7298;
			static constexpr double ACCELERATOR = 1.49618;

			canonical_rng& rng = *rng_ptr;
			return INERTIA * vi
				+ ACCELERATOR * rng() * (pbest - xi)
				+ ACCELERATOR * rng() * (lbest - xi);
		};

		particle& p = particles[idx];
		const vec_t& lbest =
			(0 == lbest_var.index())
			? *std::get<0>(lbest_var) // variant holds `const vec_t*`
			: *std::get<1>(lbest_var); // variant holds `buffer_t::viewer`

		for (size_t d = 0; d < dimension; ++d) {
			double& vi = p.velocity[d];
			double& xi = p.position[d];
			vi = calculate_velocity(vi, xi, p.best_position[d], lbest[d]);
			xi += vi;

			// Confinement
			if (xi < min) {
				xi = min;
				vi = 0;
			}
			else if (xi > max) {
				xi = max;
				vi = 0;
			}
			else {
				// nothing happens
			}
		}
	}

	range_t make_iteration_range(size_t first) {
		return { first
			   , std::min(first + iteration_per_task, iteration) };
	}

	/*
	auto fork(const range_t& subswarm_range, const range_t& iteration_range, canonical_rng* rng_ptr) {
		return[this
			, tracer = fork_tracer(this)
			, subswarm_range, iteration_range
			, rng_ptr] (worker_handle& wh) {
			pso_main_loop(subswarm_range, iteration_range, rng_ptr, wh);
		};
	}
	*/
	void pso_main_loop(range_t subswarm_range, range_t iteration_range, canonical_rng* rng_ptr) {
		//std::cout << "1";
		omp_set_num_threads(7);
		// Loop
		while (1) {
//#pragma omp parallel for
			for (int i = iteration_range.first; i < iteration_range.second; ++i) {
//#pragma omp parallel for
				for (int j = subswarm_range.first; j < subswarm_range.second; ++j) {
					// Lbest				
					// const vec_t& lbest = get_lbest_unsafe(j);
					var_t lbest_var = get_lbest(j, subswarm_range);
					// Update velocity, position				
					move_particle(j, std::move(lbest_var), rng_ptr); // Sink

					evaluate_particle(j);
					std::cout << "OK" << std::endl;
				} // end of particle

#ifdef PAPSO2_TRACK_CONVERGENCY
				// Only one subswarm would periodly update, print global best
				// Here the first subswarm is chosen
				if (0 == subswarm_range.first
					&& (i + 1) % 100 == 0) {
					auto gbest = update_gbest();
					printf("%6.4lf ", gbest.best_value);
				}
#endif
			} // end of iteration

			// Fork next iterations
			if (iteration_range.second < iteration) {
				range_t next_iter_range = make_iteration_range(iteration_range.second);
				continue;
				//wh.execute(fork(subswarm_range, next_iter_range, rng_ptr));
			}
			else
				break;
		}
	}

public:

	class papso_result_t {
		std::unique_ptr<basic_papso> state_;
	public:
		papso_result_t(std::unique_ptr<basic_papso> state)
			: state_(std::move(state)) {}
		papso_result_t(papso_result_t&& oth) noexcept
			: state_(std::move(oth.state_)) {}
		papso_result_t& operator= (papso_result_t&& rhs) noexcept {
			state_ = std::move(rhs.state_);
			return *this;
		}

		// Block until finished
		std::tuple<double, vec_t> get() {
			auto& state = *state_;
			auto check_for_completion = [&]() {
				bool is_completed = (0 == state.forks);
				return is_completed;
			};

			// Wait for finish
			{
				std::unique_lock lock{ state.completion_mtx };
				state.completion_cv.wait(lock, check_for_completion);
			}

			// Get result
			auto& gbest = state.update_gbest();
			double best_value = gbest.best_value;
			vec_t best_position = std::move(gbest.best_position);
			state_.reset(); // Release resource
			return { best_value, std::move(best_position) };
		}
	};

	static auto parallel_async_pso(size_t iter_per_task, const optimization_problem_t& problem) {
		auto pso_state_uptr = std::make_unique<basic_papso>(problem.function, problem.feasible_bound, problem.dimension, iter_per_task);
		auto& state = *pso_state_uptr;

		//using worker_handle = hungbiu::hb_executor::worker_handle;

		// Initialize
		/*
		auto remainder = swarm_size % fork_count;
		if (remainder) {
			++fork_count;
		}
		*/
		state.initialize_state(7);
		state.initialize_swarm(state.rngs[0]);

		// Forks
		/*
		size_t fork_size = swarm_size / fork_count;
		for (size_t i = 0; i < fork_count; ++i) {
			range_t subswarm_range;
			subswarm_range.first = fork_size * i;
			subswarm_range.second = (i + 1) * fork_size;
			subswarm_range.second = std::min<size_t>(subswarm_range.second, swarm_size);

			range_t iter_range = state.make_iteration_range(0);

			etor.execute(state.fork(subswarm_range, iter_range, &state.rngs[i]));
		}
		*/
		range_t subswarm_range;
		subswarm_range.first = 0;
		subswarm_range.second = swarm_size;
		range_t iter_range;
		iter_range.first = 0;
		iter_range.second = iteration;
		std::cout << "OK" << std::endl;
		state.pso_main_loop(subswarm_range, iter_range,&state.rngs[0]);
		return basic_papso::papso_result_t{ std::move(pso_state_uptr) };
		
	}
};

//using papso = basic_papso<hungbiu::spmc_buffer<vec_t>, 2, 40, 5000>;

#endif
