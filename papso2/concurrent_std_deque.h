#ifndef _CONCURRENT_STD_DEQUE
#define _CONCURRENT_STD_DEQUE
#include <deque>
#include <mutex>
#include <atomic>
namespace hungbiu
{
	template <typename T>
	class concurrent_std_deque
	{
		using lock_t = std::mutex;
		lock_t alignas(64) lock_;
		std::deque<T> deque_;
		std::atomic<std::size_t> size_{ 0 };

	public:
		concurrent_std_deque() = default;
		~concurrent_std_deque() = default;
		concurrent_std_deque(concurrent_std_deque&& oth) noexcept 
		{
			std::lock_guard lg{ oth.lock_ };
			deque_ = std::move(oth.deque_);
		}
		concurrent_std_deque& operator=(concurrent_std_deque&& rhs) noexcept
		{
			if (this != &rhs) {
				std::lock_guard slk{ lock_, rhs.lock_ };
				deque_ = std::move(rhs.deque_);
				size_ = rhs.size_.load();
				rhs.size_.store(0);
			}
			return *this;
		}

		void push_back(T& v)
		{
			std::lock_guard lg{ lock_ };
			deque_.emplace_back(std::move(v));
			size_.fetch_add(1, std::memory_order_relaxed);
		}
		void push_front(T& v)
		{
			std::lock_guard lg{ lock_ };
			deque_.emplace_front(std::move(v));
			size_.fetch_add(1, std::memory_order_relaxed);
		}
		[[nodiscard]] bool pop_front(T& v) noexcept
		{
			std::lock_guard lg{ lock_ };
			if (deque_.empty()) return false;
			v = std::move(deque_.front());
			deque_.pop_front();
			size_.fetch_sub(1, std::memory_order_relaxed);
			return true;
		}
		[[nodiscard]] bool pop_back(T& v) noexcept
		{
			std::lock_guard lg{ lock_ };
			if (deque_.empty()) return false;
			v = std::move(deque_.back());
			deque_.pop_back();
			size_.fetch_sub(1, std::memory_order_relaxed);
			return true;
		}
	};
} // end namespace hungbiu

#endif // _CONCURRENT_STD_DEQUE