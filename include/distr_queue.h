#pragma once

#include <memory>
#include <type_traits>

#include "host_object.h"
#include "runtime.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	template <typename CGF>
	constexpr bool is_safe_cgf = std::is_standard_layout<CGF>::value;

	struct capture_exfiltrator {
	  public:
		template <typename Capture>
		static auto exfiltrate_by_copy(const Capture& cap) {
			return cap.exfiltrate_by_copy();
		}

		template <typename Capture>
		static auto exfiltrate_by_move(const Capture& cap) {
			return cap.exfiltrate_by_move();
		}

		template <typename Captures>
		static auto exfiltrate_tuple_by_copy(const Captures& caps) {
			return exfiltrate_tuple_by_copy(caps, std::make_index_sequence<std::tuple_size_v<Captures>>{});
		}

		template <typename Captures>
		static auto exfiltrate_tuple_by_move(const Captures& caps) {
			return exfiltrate_tuple_by_move(caps, std::make_index_sequence<std::tuple_size_v<Captures>>{});
		}

	  private:
		template <typename Captures, size_t... Is>
		static auto exfiltrate_tuple_by_copy(const Captures& caps, std::index_sequence<Is...>) {
			return std::tuple{std::get<Is>(caps).exfiltrate_by_copy()...};
		}

		template <typename Captures, size_t... Is>
		static auto exfiltrate_tuple_by_move(const Captures& caps, std::index_sequence<Is...>) {
			return std::tuple{std::get<Is>(caps).exfiltrate_by_move()...};
		}
	};

} // namespace detail

class distr_queue;

struct allow_by_ref_t {};

inline constexpr allow_by_ref_t allow_by_ref{};

namespace experimental {

	template <typename>
	class capture;

	template <typename T, int Dims>
	class buffer_data {
	  public:
		buffer_data() : data{0, detail::zero_range} {}

		explicit operator bool() const { return data.get_pointer() != nullptr; }

		celerity::range<Dims> get_range() const { return detail::range_cast<Dims>(data.get_range()); }
		const T* get_pointer() const { return static_cast<T>(data.get_pointer()); }

	  private:
		friend class detail::capture_exfiltrator;

		detail::raw_buffer_data data;

		explicit buffer_data(detail::raw_buffer_data raw) : data{std::move(raw)} { assert(data.get_size() / data.get_range().size() == sizeof(T)); }
	};

	template <typename T, int Dims>
	class capture<buffer<T, Dims>> {
	  public:
		using value_type = buffer_data<T, Dims>;

		explicit capture(const buffer<T, Dims>& buf) : bid{detail::get_buffer_id(buf)}, sr{{{}, buf.get_range()}} {}
		explicit capture(const buffer<T, Dims>& buf, const subrange<Dims>& sr) : bid{detail::get_buffer_id(buf)}, sr{sr} {}

	  private:
		friend class detail::capture_exfiltrator;

		detail::buffer_id bid;
		subrange<Dims> sr;

		value_type exfiltrate_by_copy() const {
			auto& bm = detail::runtime::get_instance().get_buffer_manager();
			return value_type{bm.get_buffer_data(bid, sr.offset, sr.range)};
		}

		value_type exfiltrate_by_move() const { return exfiltrate_by_copy(); }
	};

	template <typename T>
	class capture<host_object<T>> {
	  public:
		static_assert(std::is_object_v<T>);

		using value_type = T;

		explicit capture(host_object<T> ho) : ho{std::move(ho)} {}

	  private:
		friend class celerity::distr_queue;

		host_object<T> ho;

		value_type exfiltrate_by_copy() const { return value_type{std::as_const(*ho.get_object())}; }

		value_type exfiltrate_by_move() const { return value_type{std::move(*ho.get_object())}; }
	};

} // namespace experimental

class distr_queue {
  public:
	distr_queue() { init(nullptr); }
	distr_queue(cl::sycl::device& device) {
		if(detail::runtime::is_initialized()) { throw std::runtime_error("Passing explicit device not possible, runtime has already been initialized."); }
		init(&device);
	}

	distr_queue(const distr_queue&) = delete;
	distr_queue& operator=(const distr_queue&) = delete;

	~distr_queue() { detail::runtime::get_instance().shutdown(); }

	/**
	 * Submits a command group to the queue.
	 *
	 * Invoke via `q.submit(celerity::allow_by_ref, [&](celerity::handler &cgh) {...})`.
	 *
	 * With this overload, CGF may capture by-reference. This may lead to lifetime issues with asynchronous execution, so using the `submit(cgf)` overload is
	 * preferred in the common case.
	 */
	template <typename CGF>
	void submit(allow_by_ref_t, CGF cgf) {
		// (Note while this function could be made static, it must not be! Otherwise we can't be sure the runtime has been initialized.)
		detail::runtime::get_instance().get_task_manager().create_task(std::move(cgf));
	}

	/**
	 * Submits a command group to the queue.
	 *
	 * CGF must not capture by reference. This is a conservative safety check to avoid lifetime issues when command groups are executed asynchronously.
	 *
	 * If you know what you are doing, you can use the `allow_by_ref` overload of `submit` to bypass this check.
	 */
	template <typename CGF>
	void submit(CGF cgf) {
		static_assert(detail::is_safe_cgf<CGF>, "The provided command group function is not multi-pass execution safe. Please make sure to only capture by "
		                                        "value. If you know what you're doing, use submit(celerity::allow_by_ref, ...).");
		submit(allow_by_ref, std::move(cgf));
	}

	/**
	 * @brief Fully syncs the entire system.
	 *
	 * This function is intended for incremental development and debugging.
	 * In production, it should only be used at very coarse granularity (second scale).
	 * @warning { This is very slow, as it drains all queues and synchronizes accross the entire cluster. }
	 */
	void slow_full_sync() { detail::runtime::get_instance().sync(); }

	template <typename T>
	typename experimental::capture<T>::value_type slow_full_sync(const experimental::capture<T>& cap) {
		// TODO schedule transfers
		slow_full_sync();
		return detail::capture_exfiltrator::exfiltrate_by_copy(cap);
	}

	template <typename... Ts>
	std::tuple<typename experimental::capture<Ts>::value_type...> slow_full_sync(const std::tuple<experimental::capture<Ts>...>& caps) {
		// TODO schedule transfers
		slow_full_sync();
		return detail::capture_exfiltrator::exfiltrate_tuple_by_copy(caps);
	}

  private:
	void init(cl::sycl::device* user_device) {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr, user_device); }
		try {
			detail::runtime::get_instance().startup();
		} catch(detail::runtime_already_started_error&) {
			throw std::runtime_error("Only one celerity::distr_queue can be created per process (but it can be copied!)");
		}
	}
};

namespace experimental {

	void drain(distr_queue&&) { detail::runtime::get_instance().shutdown(); }

	template <typename T>
	typename experimental::capture<T>::value_type drain(distr_queue&& q, const experimental::capture<T>& cap) {
		// TODO schedule transfers
		drain(std::move(q));
		return detail::capture_exfiltrator::exfiltrate_by_move(cap);
	}

	template <typename... Ts>
	std::tuple<typename experimental::capture<Ts>::value_type...> drain(distr_queue&& q, const std::tuple<experimental::capture<Ts>...>& caps) {
		// TODO schedule transfers
		drain(std::move(q));
		return detail::capture_exfiltrator::exfiltrate_tuple_by_move(caps);
	}

} // namespace experimental
} // namespace celerity
