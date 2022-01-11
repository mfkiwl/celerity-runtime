#pragma once

#include "runtime.h"


namespace celerity::experimental {

template <typename T>
class host_object;

} // namespace celerity::experimental

namespace celerity::detail {

struct host_object_info {
	std::optional<task_id> last_writer;
};

class host_object_manager {
  public:
	host_object_id create_host_object() {
		const std::lock_guard lock{mutex};
		const auto id = next_id++;
		objects.emplace(id, host_object_info{});
		return id;
	}

	void destroy_host_object(const host_object_id id) {
		const std::lock_guard lock{mutex};
		objects.erase(id);
	}

	// true-result only reliable if no calls to create_host_object() are pending
	bool has_active_objects() const {
		const std::lock_guard lock{mutex};
		return !objects.empty();
	}

  private:
	mutable std::mutex mutex;
	host_object_id next_id = 0;
	std::unordered_map<host_object_id, host_object_info> objects;
};

// Base for `state` structs in all host_object specializations: registers and unregisters host_objects with the host_object_manager.
struct host_object_tracker {
	detail::host_object_id id;

	host_object_tracker() {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr); }
		id = detail::runtime::get_instance().get_host_object_manager().create_host_object();
	}

	host_object_tracker(host_object_tracker&&) = delete;
	host_object_tracker& operator=(host_object_tracker&&) = delete;

	~host_object_tracker() { detail::runtime::get_instance().get_host_object_manager().destroy_host_object(id); }
};

} // namespace celerity::detail

namespace celerity::experimental {

template <typename T>
class host_object {
  public:
	host_object() : shared_state{std::make_shared<state>(std::in_place)} {}

	explicit host_object(const T& obj) : shared_state{std::make_shared<state>(std::in_place, obj)} {}

	explicit host_object(T&& obj) : shared_state{std::make_shared<state>(std::in_place, std::move(obj))} {}

	template <typename... CtorParams>
	explicit host_object(const std::in_place_t, CtorParams&&... ctor_args)
	    : shared_state{std::make_shared<state>(std::in_place, std::forward<CtorParams>(ctor_args)...)} {}

  private:
	template <typename, side_effect_order>
	friend class side_effect;

	struct state : detail::host_object_tracker {
		T object;

		template <typename... CtorParams>
		explicit state(const std::in_place_t, CtorParams&&... ctor_args) : object{std::forward<CtorParams>(ctor_args)...} {}
	};

	detail::host_object_id get_id() const { return shared_state->id; }
	T* get_object() const { return &shared_state->object; }

	std::shared_ptr<state> shared_state;
};

template <typename T>
explicit host_object(T& obj) -> host_object<std::remove_const_t<T>>;

template <typename T>
class host_object<T&> {
  public:
	explicit host_object(T& obj) : shared_state{std::make_shared<state>(obj)} {}

	explicit host_object(std::reference_wrapper<T> ref) : shared_state{std::make_shared<state>(ref.get())} {}

  private:
	template <typename, side_effect_order>
	friend class side_effect;

	struct state : detail::host_object_tracker {
		T& object;

		explicit state(T& object) : object{object} {}
	};

	detail::host_object_id get_id() const { return shared_state->id; }
	T* get_object() const { return &shared_state->object; }

	std::shared_ptr<state> shared_state;
};

template <typename T>
explicit host_object(std::reference_wrapper<T> ref) -> host_object<T&>;

template <>
class host_object<void> {
  public:
	explicit host_object() : shared_state{std::make_shared<state>()} {}

  private:
	template <typename, side_effect_order>
	friend class side_effect;

	struct state : detail::host_object_tracker {};

	detail::host_object_id get_id() const { return shared_state->id; }

	std::shared_ptr<state> shared_state;
};

explicit host_object() -> host_object<void>;

} // namespace celerity::experimental
