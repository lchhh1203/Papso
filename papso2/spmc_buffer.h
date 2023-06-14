#ifndef _SMPC_BUFFER
#define _SMPC_BUFFER
#define DEBUG_PRINT 0
#include <atomic>
#include <array>
#include <mutex>
#include <shared_mutex>
#if DEBUG_PRINT
#include <stdio.h>
#endif
// Ì×ÉÏËø£¡
// ²âÊÔ£¡

namespace hungbiu {
	// For:
	// 1) Single writer that might update at an arbitrary frequency
	// 2) Multiple readers that might take an arbitrary period of time;
	// Requires T to be default constructible
	template <typename T, size_t Associativity = 4>
	requires (Associativity >= 2)
	class spmc_buffer {
		// count == -1, owned exclusively
		// count == 0, nobody is using
		// count > 0, shared for reading
		using counter_type = std::atomic<int>;
		struct slot_type {
			counter_type counter alignas(64) = { 0 };
			T value alignas(64) = {};
		};

		class lock_base {
		protected:
			spmc_buffer* pbuffer_;
			counter_type* pcounter_;
		public:
			lock_base(spmc_buffer* pb, counter_type* pc) :
				pbuffer_(pb), pcounter_(pc) {}
			lock_base(lock_base&& oth) noexcept :
				pbuffer_(std::exchange(oth.pbuffer_, nullptr))
				, pcounter_(std::exchange(oth.pcounter_, nullptr)) {}
			virtual ~lock_base() {}
			lock_base(const lock_base&) = delete;
			lock_base& operator=(const lock_base&) = delete;

			explicit operator bool() const noexcept {
				return owns_lock();
			}
			bool owns_lock() const noexcept {
				return pbuffer_ && pcounter_;
			}
		};

		class read_lock : public lock_base{
			using lock_base::pbuffer_;
			using lock_base::pcounter_;
			void release() {
				if (owns_lock()) {
					pbuffer_->release_read(pcounter_);
				}
			}
		public:
			read_lock(spmc_buffer* pb, counter_type* pc) :
				lock_base(pb, pc) {}
			read_lock(read_lock&& oth) noexcept :
				lock_base(std::move(oth)) {}
			~read_lock() override {
				release();
			}
			void unlock() {
				release();
				pbuffer_ = nullptr;
				pcounter_ = nullptr;
			}
			using lock_base::owns_lock;
			using lock_base::operator bool;
		};

		class write_lock : public lock_base {
			using lock_base::pbuffer_;
			using lock_base::pcounter_;
			size_t write_idx_;
		public:
			write_lock(spmc_buffer* pb, counter_type* pc, size_t widx) : 
				lock_base(pb, pc), write_idx_(widx) {}
			write_lock(write_lock&& oth) noexcept :
				lock_base(std::move(oth))
				, write_idx_(oth.write_idx_) {}
			~write_lock() {
				if (owns_lock()) {
					pbuffer_->release_write(pcounter_);
				}
			}
			using lock_base::owns_lock;
			using lock_base::operator bool;
			size_t write_idx() const noexcept {
				return write_idx_;
			}
		};
	
	public:
		class viewer {
			const T* pval_;
			read_lock rlock_;

			const T* get() const noexcept {
				return pval_;
			}

		public:
			viewer(read_lock rlock, const T* pval) :
				pval_(pval), rlock_(std::move(rlock)) {}
			viewer(viewer&& oth) noexcept :
				pval_(oth.pval_), rlock_(std::move(oth.rlock_)) {}
			~viewer() {}

			void unlock() {
				if (rlock_.owns_lock()) {
					rlock_.unlock();
				}
			}
			const T& operator*() const noexcept { return *get(); }
			const T* operator->() const noexcept { return get(); }
		};
		
	private:
		std::atomic<T*> pending_value_ = { nullptr }; // Non-const for buffer reuse
		std::atomic<size_t> read_index_ = { 0 };
		std::array<slot_type, Associativity> buffers_;

		std::pair<counter_type*, const T*> acquire_read() noexcept {
			// Acquire index of the latest value
			size_t read_idx = read_index_.load(std::memory_order_acquire) % Associativity;
			counter_type* pcounter = &buffers_[read_idx].counter;

			// Read_index are published after releasing write lock 
			// enabling non-blocking read
			pcounter->fetch_add(1, std::memory_order_acq_rel);

			return { pcounter, &buffers_[read_idx].value };
		}

		void release_read(counter_type* pcounter) noexcept {
			// Decrement rwlock (read count)
			int read_count = pcounter->fetch_sub(1, std::memory_order_acq_rel);
			proceed_pending_write();
		}

		bool acquire_write_cas(int& read_count, counter_type* pcounter) {
			return pcounter->compare_exchange_strong(read_count, -1, std::memory_order_acq_rel);
		}

		bool acquire_write_ptr(counter_type* pcounter) {
			int read_count = pcounter->load(std::memory_order_acquire);
			if (0 == read_count) { // Try to acquire write if no reader
				bool b = acquire_write_cas(read_count, pcounter);

#if DEBUG_PRINT
				printf("acquire_write_cas(): actual = %d\n", read_count);
#endif
				return b;
			}
			else {
				return false;
			}
		}

		// Acquire write lock at (read_index_ + 1)
		std::pair<counter_type*, size_t> acquire_write_next() {
			size_t read_idx = read_index_.load(std::memory_order_acquire);
			size_t write_idx = (read_idx + 1) % Associativity;
			counter_type* pcounter = &buffers_[write_idx].counter;
			if (acquire_write_ptr(pcounter)) {
				return { pcounter, write_idx };
			}
			else {
				return { nullptr, write_idx };
			}
		}
		
		// Acquire write lock from any available postion
		std::pair<counter_type*, size_t> acquire_write() noexcept {
#if DEBUG_PRINT
			printf("acquire_write():\n");
#endif

			// Traverse every slot in the buffers_ and find a position to write to
			size_t read_idx = read_index_.load(std::memory_order_acquire);
			size_t write_idx = read_idx + 1;
			for (; write_idx % Associativity != read_idx; ++write_idx) {

#if DEBUG_PRINT
				printf("\twrite_index: %llu\n", write_idx);
#endif

				counter_type* pcounter = &buffers_[write_idx % Associativity].counter;
				if (acquire_write_ptr(pcounter)) {
					return { pcounter, write_idx % Associativity };
				}
			}

			return { nullptr, write_idx % Associativity };
		}

		void release_write(counter_type* pcounter) {
			pcounter->store(0, std::memory_order_release);
		}
		
		write_lock get_write_lock_next() noexcept {
			auto [pc, widx] = acquire_write_next();
			return { this, pc, widx };
		}

		write_lock get_write_lock() noexcept {
			auto [pc, widx] = acquire_write();
			return { this, pc, widx };
		}	

		template <typename U>
		void add_pending_write(U&& val, T* ptr) {			
			T* pnew = ptr 
						? new(ptr) T(std::forward<U>(val))  // Construct in-place
						: new T(std::forward<U>(val));		// Allocate to construct
			T* pold = pending_value_.exchange(pnew, std::memory_order_acq_rel);
		}

		// if (buffer[read_idx + 1].count == 0 && pending_update) 
		//		if (lock(write)) 
		//			write buffer
		//			bump read_idx
		//		else return
		// else return
		void proceed_pending_write() noexcept {
			// Check if there is any pending write
			if (nullptr == pending_value_.load(std::memory_order_acquire)) {
				return;
			}
			const T* pval = pending_value_.exchange(nullptr, std::memory_order_acq_rel);
			if (!pval) {
				return;
			}

#if DEBUG_PRINT
			printf("proceed_pending_write(): ");
#endif
			{
				// Check if the buffer, where to be written, is available
				// Break if it's still being read/written by another thread
				write_lock wlock = get_write_lock_next();
				if (!wlock) {

#if DEBUG_PRINT
					printf("failed to get write lock next\n");
#endif
					return;
				}

#if DEBUG_PRINT
				printf("write to next\n");
#endif
				buffers_[wlock.write_idx()].value = std::move(*pval);
			}

			// Publish new value
			read_index_.fetch_add(1, std::memory_order_acq_rel);

			// Clear
			delete pval;
		}

	public:
		spmc_buffer() {}
		spmc_buffer(spmc_buffer&& oth) noexcept {
			auto p = oth.pending_value_.exchange(nullptr);
			if (!p) {
				auto idx = oth.read_index_.load();
				slot_type& slot = oth.buffers_[idx];
				counter_type* pc = &(slot.counter);
				write_lock wlock_oth{ &oth, pc, idx };
				buffers_[0].value = std::move(slot.value);
			}
			else {
				buffers_[0].value = std::move(*p);
				delete p;
			}
		}
		~spmc_buffer() {}

		// Single writer
		template <typename U>
		void put(U&& val) {
			// Retrieve if there is a pending write
			T* old = pending_value_.load(std::memory_order_acquire);
			if (old) {
				old = pending_value_.exchange(nullptr);
			}

#if DEBUG_PRINT
			printf("put(): ");
#endif

			size_t write_idx = 0;
			{
				write_lock wlock = get_write_lock();

				auto wi = (read_index_.load() + 1) % Associativity;
#if DEBUG_PRINT
				printf("next pos: %llu, read count: %d, "
					,  wi
					, buffers_[wi].counter.load());
#endif
				if (!wlock) {
#if DEBUG_PRINT
					printf("add to pending write\n");
#endif
					add_pending_write(std::forward<U>(val), old);
					return;
				}
#if DEBUG_PRINT
				printf("direct write\n");
#endif
				write_idx = wlock.write_idx();
				buffers_[write_idx].value = std::forward<U>(val);
			}
			
			// Publish new value
			read_index_.store(write_idx, std::memory_order_release);
		}

		viewer get() noexcept {
			auto [pcounter, pval] = acquire_read();
			read_lock rlock = { this, pcounter };
			return viewer{ std::move(rlock), pval };
		}
	};


	template <typename T>
	class naive_spmc_buffer { // T does not need to be aligned!
		std::shared_mutex smtx_ alignas(64) = {};
		T val_ alignas(64) = {};
	public:
		class viewer {
			std::shared_lock<std::shared_mutex> slock_;
			const T* pv_;
		public:
			viewer(std::shared_mutex& smtx, const T* p) :
				slock_(smtx), pv_(p) {}
			viewer(viewer&& oth) noexcept :
				slock_(std::move(oth.slock_))
				, pv_(std::exchange(oth.pv_, nullptr)) {}
			viewer() {}

			void unlock() {
				if (slock_.owns_lock()) {
					slock_.unlock();
				}
				pv_ = nullptr;
			}
			const T& operator*() const noexcept { return *pv_; }
			const T* operator->() const noexcept { return pv_; }
			bool owns_lock() const noexcept { return slock_.owns_lock(); }
			explicit operator bool() const noexcept { return slock_.owns_lock(); }
		};
	
		naive_spmc_buffer() {}
		naive_spmc_buffer(naive_spmc_buffer&& oth) noexcept {
			std::unique_lock ulock{ oth.smtx_ };
			val_ = std::move(oth.val_);
		}
		~naive_spmc_buffer() {}
		

		viewer get() noexcept {
			return { smtx_, &val_ };
		}
		template <typename U>
		void put(U&& val) {
			std::lock_guard guard{ smtx_ };
			val_ = std::forward<U>(val);
		}
	};
}
#endif 