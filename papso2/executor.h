#ifndef _EXECUTOR
#define _EXECUTOR
#include <memory>
#include <type_traits>
#include <vector>
#include <cstddef>
#include <random>
#include <concepts>
#include <future>
#include <condition_variable>
#include <mutex>
#include <thread>
#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#include <processthreadsapi.h>
#include <winbase.h>
#endif
#include "concurrent_std_deque.h"

// lazy spin up + cv
namespace hungbiu
{	
	class hb_executor
	{			
	public:	// Template aliases used by hb_executor
		template <typename T>
		using promise_t = std::promise<T>;
		template <typename T>
		using future_t = std::future<T>;
		class worker_handle; // Forward declaration

	private:
		// r_task_wrapper: provide aysnc result
		template <typename R>
		using r_task_wrapper = std::packaged_task<R(worker_handle&)>;
		template <typename F, typename R>
		static r_task_wrapper<R> make_task(F&& func)
		{
			using F_decay = std::decay_t<F>;
			return r_task_wrapper<R>(std::forward<F_decay>(func));
		}

		// task_wrapper: does not provide async result & SSO
		struct task_wrapper_concept
		{
			void (*_destructor)(void*) noexcept;
			void (*_move)(void*, void*) noexcept;
			void (*_run)(void*, worker_handle&);
		};

		static constexpr auto small_size = sizeof(void*) * 7;
		template <typename T>
		static constexpr bool is_small_object() { return sizeof(T) <= small_size; }

		template <typename T, bool Small = is_small_object<T>()>
		struct task_wrapper_model;
		template <typename F>
		struct task_wrapper_model<F, true>
		{
			F task_;

			template <typename U>
			task_wrapper_model(U&& func) :
				task_(std::forward<F>(func)) {}
			task_wrapper_model(task_wrapper_model&& oth) :
				task_(std::move(oth.task_)) {}
			~task_wrapper_model() = default;

			static void _destructor(void* p) noexcept
			{
				static_cast<task_wrapper_model*>(p)->~task_wrapper_model();
			}
			static void _move(void* lhs, void* rhs) noexcept
			{
				auto& model = *static_cast<task_wrapper_model*>(rhs);
				new (lhs) task_wrapper_model(std::move(model));
			}
			static void _run(void* p, worker_handle& h)
			{
				auto& t = static_cast<task_wrapper_model*>(p)->task_;
				std::invoke(t, h);
			}
			static constexpr task_wrapper_concept vtable_ = { _destructor, _move, _run };
		};
		template <typename F>
		struct task_wrapper_model<F, false>
		{
			std::unique_ptr<F> ptask_;

			template <typename U>
			task_wrapper_model(U&& func) :
				ptask_(std::make_unique<F>(std::forward<F>(func))) {}
			task_wrapper_model(task_wrapper_model&& oth) :
				ptask_(std::move(oth.ptask_)) {}
			~task_wrapper_model() = default;

			static void _destructor(void* p) noexcept
			{
				static_cast<task_wrapper_model*>(p)->~task_wrapper_model();
			}
			static void _move(void* lhs, void* rhs) noexcept
			{
				auto& model = *static_cast<task_wrapper_model*>(rhs);
				new (lhs) task_wrapper_model(std::move(model));
			}
			static void _run(void* p, worker_handle& h)
			{
				auto pt = static_cast<task_wrapper_model*>(p)->ptask_.get();
				if (pt) std::invoke(*pt, h);
			}
			static constexpr task_wrapper_concept vtable_ = { _destructor, _move, _run };
		};

		class task_wrapper
		{
			// Constness: the vtable is a const object!
			const task_wrapper_concept* task_vtable_{ nullptr };
			std::aligned_storage_t<small_size> task_;
		public:
			task_wrapper() : task_() {}
			template <typename T>
			task_wrapper(T&& t)
			{
				using DecayT = std::decay_t<T>;
				using model_t = task_wrapper_model<DecayT>;
				task_vtable_ = &model_t::vtable_;
				new (&task_) model_t(std::forward<DecayT>(t));
			}
			task_wrapper(task_wrapper&& oth) noexcept :
				task_vtable_(oth.task_vtable_)
			{
				task_vtable_->_move(&task_, &oth.task_);
			}
			~task_wrapper()
			{
				if (task_vtable_) task_vtable_->_destructor(&task_);
			}
			task_wrapper& operator=(task_wrapper&& rhs) noexcept
			{
				if (this != &rhs) {
					// Destroy current task if there is one
					if (task_vtable_) task_vtable_->_destructor(&task_);

					// Copy rhs's vtable BEFORE moving the rhs's task
					// because rhs's task has an arbitrary type
					task_vtable_ = std::exchange(rhs.task_vtable_, nullptr);
					task_vtable_->_move(&task_, &rhs.task_);
				}
				return *this;
			}

			bool valid() const noexcept
			{
				return task_vtable_;
			}
			explicit operator bool() const noexcept
			{
				return valid();
			}
			void run(worker_handle& h)
			{
				if (task_vtable_) task_vtable_->_run(&task_, h);
			}
		};

		class worker; // Forward declaration because worker::get_handle() return a woker_handle object;
					  // Have to be PRIVATE
	public:		
		// --------------------------------------------------------------------------------
		// worker_handler
		// used by task object to submit work
		// --------------------------------------------------------------------------------		
		class worker_handle
		{
			worker* ptr_worker_;
		public:
			worker_handle(worker* worker) noexcept :
				ptr_worker_(worker) {}
			worker_handle(const worker_handle& oth) noexcept :
				ptr_worker_(oth.ptr_worker_) {}
			worker_handle& operator=(const worker_handle& rhs) noexcept
			{
				if (this != &rhs) ptr_worker_ = rhs.ptr_worker_;
				return *this;
			}

			template <typename T>
			static bool future_ready(const std::future<T>& fut)
			{
				const auto status = fut.wait_for(std::chrono::nanoseconds{ 0 });
				return std::future_status::ready == status;
			}

			// Suspend current task to execute others
			template <typename R>
			R get(future_t<R>& fut)
			{
				while (!future_ready(fut)) {
					task_wrapper tw{};
					if (ptr_worker_->_pop(tw) ||
						ptr_worker_->_steal(tw)) {
						tw.run(*this);
					}
				}
				return fut.get();
			}

			// Submit a task to current thread, return future to obtain result
			template <
				typename F
				, typename R = std::invoke_result_t<F, worker_handle&> >
			requires std::invocable<F, hb_executor::worker_handle&>
				[[nodiscard]] future_t<R> execute_return(F&& func) const
			{
				auto t = make_task<F, R>(std::forward<F>(func));
				auto fut = t.get_future();
				ptr_worker_->_push(std::move(t));
				return fut;
			}

			// Submit a task to current thread and doesn't return result
			template <typename F>
			requires std::invocable<F, hb_executor::worker_handle&>
			void execute(F&& func) const
			{
				ptr_worker_->_push(std::move(func));
			}
		}; // end of class worker handle
	
	private:
		using rng_t = std::default_random_engine;
		// --------------------------------------------------------------------------------
		// A worker object is the working context of an OS thread
		// --------------------------------------------------------------------------------
		class alignas(64) worker
		{
			friend class worker_handle;
			template <typename T>
			using deque_t = concurrent_std_deque<T>;//concurrent_std_deque<T>;

			hb_executor* etor_;
			std::size_t index_;
			deque_t<task_wrapper> run_stack_;
			std::condition_variable_any cv_;
			std::mutex mtx_; // use this mutex to wait for condition

			// Waiting, Ready, Running
			// `Waiting` -> `Ready`:   Dispatcher makes a worker `Ready` after assigning task to it;
			// `Ready` -> `Running`:   Worker makes itself run when resume from waiting on cv
			// `Running` -> `Waiting`: Worker waits on cv if it can't find any task
			//static constexpr unsigned Waiting = 0b0001; // Waiting for task
			//static constexpr unsigned Pending = 0b0010; // Has pending task
			//static constexpr unsigned Running = 0b0100; // Executing task

			alignas(64) unsigned pending_{ 0 };
			rng_t rng_;

			// Push a forked task onto stack
			void _push(task_wrapper tw)
			{
				run_stack_.push_back(tw); // Notify one?
			}
			// Pop a task from stack for the worker itself to execute
			[[nodiscard]] bool _pop(task_wrapper& tw) noexcept
			{
				return run_stack_.pop_back(tw);
			}
			[[nodiscard]] bool _steal(task_wrapper& tw)
			{
				return etor_->steal(tw, index_, &rng_);
			}
		public:
			//static constexpr auto RUN_QUEUE_SIZE = 256u;
			worker(hb_executor& etor, std::size_t idx) :
				etor_(&etor), index_(idx) {}
			~worker() {}
			worker(worker&& oth) noexcept // Should not be used, only for vector
				: etor_(std::exchange(oth.etor_, nullptr))
				, index_(std::exchange(oth.index_, -1))
				, run_stack_(std::move(oth.run_stack_))
				/*, state_(oth.state_)*/
				, rng_(std::move(oth.rng_)) {}
			worker& operator=(const worker&) = delete;

			void operator()(std::stop_token stoken)
			{
				auto h = get_handle();
				const bool enable_stealing = etor_->enable_stealing_;
				while (!etor_->is_done() && !stoken.stop_requested()) {
					// This task wrapper must be destroyed at the end of the loop
					task_wrapper tw;

					// get work from local stack 
					if (_pop(tw)) {
						tw.run(h);
						continue;
					}

					// steal from others
					if (enable_stealing) {
						if (etor_->steal(tw, index_, &rng_)) {
							tw.run(h);
							continue;
						}
					}

					// Give up time slice
					std::this_thread::yield();
				} // End of while loop
			}
			void assign(task_wrapper& tw)
			{
				run_stack_.push_front(tw);
			}
			[[nodiscard]] bool try_steal(task_wrapper& tw) noexcept
			{
				return run_stack_.pop_front(tw);
			}
			void notify_work() {
				{
					std::lock_guard guard{ mtx_ };
					pending_ = static_cast<unsigned>(true);
				}
				cv_.notify_one();
			}
			worker_handle get_handle() noexcept
			{
				return worker_handle{ this };
			}
		};

		// Worker thread's main function
		static auto get_thread_affinity_mask(unsigned index) {
			// Allow the thread to run on 2 logical processors to create scheduling slack
			// Actually this targets hyperthreading
			const unsigned core_count = std::thread::hardware_concurrency();
			auto my_mask = 1 << (index % core_count);
			return my_mask;
		}
		static void thread_main(std::stop_token stoken, hb_executor* this_, std::size_t init_idx)
		{
			// Pin thread to processor
#ifdef _MSC_VER
			if (!SetThreadAffinityMask(GetCurrentThread(), get_thread_affinity_mask(init_idx))) {
				throw std::exception{ "failed to set thread affinity!" };
			}
#endif
			this_->workers_[init_idx].operator()(stoken);
		}
		
		// --------------------------------------------------------------------------------
		// Data members of executor
		// --------------------------------------------------------------------------------
		mutable std::atomic<bool> is_done_{ false };
		std::atomic<size_t> ticket_{ 0 };
		std::vector<worker> workers_;
		std::vector<std::jthread> threads_;

#ifdef COUNT_STEALING
		alignas(64) std::atomic<size_t> steal_count_ { 0 };
#endif

	public:
		bool done() noexcept
		{
			if (is_done()) {
				return true;
			}

			bool done = false;
			is_done_.compare_exchange_strong(done, true, std::memory_order_acq_rel);
			return is_done();
		}
		bool is_done() const noexcept
		{
			return is_done_.load(std::memory_order_acquire);
		}
#ifdef COUNT_STEALING
		size_t get_steal_count() const noexcept {
			return steal_count_.load(std::memory_order_acquire);
		}
#endif
	private:
		// not thread-safe (single producer, multi consumers)
		std::size_t random_idx(rng_t* rng) noexcept
		{
			static thread_local std::mt19937_64 engine;
			if (rng) {
				return std::uniform_int_distribution<std::size_t>()(*rng);
			}
			else {
				return std::uniform_int_distribution<std::size_t>()(engine);
			}
		}
		void dispatch(task_wrapper tw)
		{
			auto idx = ticket_.load();
			const auto sz = workers_.size();
			workers_[idx % sz].assign(tw);
			workers_[idx % sz].notify_work();
			ticket_.compare_exchange_strong(idx, idx + 1, std::memory_order_acq_rel);
		} 
		const bool enable_stealing_;
		[[nodiscard]] bool steal(task_wrapper& tw, const std::size_t idx, rng_t* rng)
		{		
			for (size_t i = idx + 1; i < idx + workers_.size(); ++i) {
				if (workers_[i % workers_.size()].try_steal(tw)) {

#ifdef COUNT_STEALING
				steal_count_.fetch_add(1, std::memory_order_relaxed);
#endif
					return true;
				}
			}
			return false;
		}		
			
	public:				
		hb_executor(size_t parallelism, bool enable_stelaing = true) :
			enable_stealing_(enable_stelaing)
		{
			workers_.reserve(parallelism);
			threads_.reserve(parallelism);
			for (auto i = 0u; i < parallelism; ++i) {
				workers_.emplace_back(*this, i);
				threads_.emplace_back(thread_main, this, i);
			}
		}
		~hb_executor()
		{		
			done();
			for (auto& t : threads_) {
				t.request_stop();
			}			
		}

		hb_executor(const hb_executor&) = delete;
		hb_executor& operator=(const hb_executor&) = delete;

		template <typename F, typename R = std::invoke_result_t<F, worker_handle&>>
		requires std::invocable<F, hb_executor::worker_handle&>
		[[nodiscard]] future_t<R> execute_return(F&& func)
		{
			if (is_done()) {
				return future_t<R>{};
			}
			auto t = make_task<F, R>(std::forward<F>(func));
			auto fut = t.get_future();
			dispatch( std::move(t) );
			return fut;
		}
		
		// Submit a task to current thread and doesn't return result
		template <typename F>
		requires std::invocable<F, hb_executor::worker_handle&>
		void execute(F&& func)
		{
			if (is_done()) { return; }
			dispatch( std::forward<F>(func) );
		}
	};

	template <typename F>
	concept is_hb_task = std::invocable<F, hb_executor::worker_handle&>;
}

#endif