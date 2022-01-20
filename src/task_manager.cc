#include "task_manager.h"

#include "access_modes.h"
#include "logger.h"
#include "print_graph.h"

namespace celerity {
namespace detail {

	task_manager::task_manager(size_t num_collective_nodes, host_queue* queue, reduction_manager* reduction_mgr)
	    : num_collective_nodes(num_collective_nodes), queue(queue), reduction_mngr(reduction_mgr) {
		// We manually generate the first init task; milestones are used later on (see generate_task_horizon() / create_barrier_task()).
		current_init_task_id = get_new_tid();
		task_map[current_init_task_id] = task::make_nop(current_init_task_id);
	}

	void task_manager::add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized) {
		std::lock_guard<std::mutex> lock(task_mutex);
		buffers_last_writers.emplace(bid, range);
		if(host_initialized) { buffers_last_writers.at(bid).update_region(subrange_to_grid_box(subrange<3>({}, range)), current_init_task_id); }
	}

	bool task_manager::has_task(task_id tid) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		return task_map.count(tid) != 0;
	}

	// Note that we assume tasks are not modified after their initial creation, which is why
	// we don't need to worry about thread-safety after returning the task pointer.
	const task* task_manager::get_task(task_id tid) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		assert(task_map.count(tid) != 0);
		return task_map.at(tid).get();
	}

	void task_manager::print_graph(logger& graph_logger) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		if(task_map.size() < 200) {
			detail::print_graph(task_map, graph_logger);
		} else {
			graph_logger.warn("Task graph is very large ({} vertices). Skipping GraphViz output", task_map.size());
		}
	}

	void task_manager::notify_milestone_reached(task_id tid, milestone_type type) {
#ifndef NDEBUG
		{
			std::lock_guard lock{task_mutex};
			assert(task_map.count(tid) != 0);
			assert(task_map.at(tid)->get_type() == to_task_type(type));
		}
#endif

		// no locking needed - see definition of executed_horizons
		// after store to milestone_task_id_for_deletion, actual cleanup happens on new task creation
		if(type == milestone_type::HORIZON) {
			executed_horizons.push(tid);
			if(executed_horizons.size() >= horizon_deletion_lag) {
				milestone_task_id_for_deletion.store(executed_horizons.front());
				executed_horizons.pop();
			}
		} else /* type == milestone_type::BARRIER */ {
			while(!executed_horizons.empty()) {
				assert(executed_horizons.front() < tid);
				// all pending horizons will be implicitly deleted
				executed_horizons.pop();
			}
			milestone_task_id_for_deletion.store(tid);

			std::lock_guard lock{barrier_mutex};
			assert(last_executed_barrier < tid);
			last_executed_barrier = tid;
			barrier_executed.notify_all();
		}
	}

	GridRegion<3> get_requirements(task const* tsk, buffer_id bid, const std::vector<cl::sycl::access::mode> modes) {
		const auto& access_map = tsk->get_buffer_access_map();
		const subrange<3> full_range{tsk->get_global_offset(), tsk->get_global_size()};
		GridRegion<3> result;
		for(auto m : modes) {
			result = GridRegion<3>::merge(result, access_map.get_requirements_for_access(bid, m, tsk->get_dimensions(), full_range, tsk->get_global_size()));
		}
		return result;
	}

	void task_manager::compute_dependencies(task_id tid) {
		using namespace cl::sycl::access;

		const auto& tsk = task_map[tid];
		const auto& access_map = tsk->get_buffer_access_map();

		auto buffers = access_map.get_accessed_buffers();
		for(auto rid : tsk->get_reductions()) {
			assert(reduction_mngr != nullptr);
			buffers.emplace(reduction_mngr->get_reduction(rid).output_buffer_id);
		}

		for(const auto bid : buffers) {
			const auto modes = access_map.get_access_modes(bid);

			std::optional<reduction_info> reduction;
			for(auto maybe_rid : tsk->get_reductions()) {
				auto maybe_reduction = reduction_mngr->get_reduction(maybe_rid);
				if(maybe_reduction.output_buffer_id == bid) {
					if(reduction) { throw std::runtime_error(fmt::format("Multiple reductions attempt to write buffer {} in task {}", bid, tid)); }
					reduction = maybe_reduction;
				}
			}

			if(reduction && !modes.empty()) {
				throw std::runtime_error(fmt::format("Buffer {} is both required through an accessor and used as a reduction output in task {}", bid, tid));
			}

			// Determine reader dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), detail::access::mode_traits::is_consumer)
			    || (reduction.has_value() && reduction->initialize_from_buffer)) {
				auto read_requirements = get_requirements(tsk.get(), bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
				if(reduction.has_value()) { read_requirements = GridRegion<3>::merge(read_requirements, GridRegion<3>{{1, 1, 1}}); }
				const auto last_writers = buffers_last_writers.at(bid).get_region_values(read_requirements);

				for(auto& p : last_writers) {
					// This indicates that the buffer is being used for the first time by this task, or all previous tasks also only read from it.
					// A valid use case (i.e., not reading garbage) for this is when the buffer has been initialized using a host pointer.
					if(p.second == std::nullopt) continue;
					const task_id last_writer = *p.second;
					assert(task_map.count(last_writer) == 1);
					add_dependency(tsk.get(), task_map[last_writer].get(), dependency_kind::TRUE_DEP);
				}
			}

			// Update last writers and determine anti-dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), detail::access::mode_traits::is_producer) || reduction.has_value()) {
				auto write_requirements = get_requirements(tsk.get(), bid, {detail::access::producer_modes.cbegin(), detail::access::producer_modes.cend()});
				if(reduction.has_value()) { write_requirements = GridRegion<3>::merge(write_requirements, GridRegion<3>{{1, 1, 1}}); }
				assert(!write_requirements.empty() && "Task specified empty buffer range requirement. This indicates potential anti-pattern.");
				const auto last_writers = buffers_last_writers.at(bid).get_region_values(write_requirements);

				for(auto& p : last_writers) {
					if(p.second == std::nullopt) continue;
					assert(task_map.count(*p.second) == 1);
					auto& last_writer = *task_map[*p.second];

					// Determine anti-dependencies by looking at all the dependents of the last writing task
					bool has_anti_dependents = false;

					for(auto dependent : last_writer.get_dependents()) {
						if(dependent.node->get_id() == tid) {
							// This can happen
							// - if a task writes to two or more buffers with the same last writer
							// - if the task itself also needs read access to that buffer (R/W access)
							continue;
						}
						const auto dependent_read_requirements =
						    get_requirements(dependent.node, bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
						// Only add an anti-dependency if we are really writing over the region read by this task
						if(!GridRegion<3>::intersect(write_requirements, dependent_read_requirements).empty()) {
							add_dependency(tsk.get(), dependent.node, dependency_kind::ANTI_DEP);
							has_anti_dependents = true;
						}
					}

					if(!has_anti_dependents) {
						// If no intermediate consumers exist, add an anti-dependency on the last writer directly.
						// Note that unless this task is a pure producer, a true dependency will be created and this is a no-op.
						// While it might not always make total sense to have anti-dependencies between (pure) producers without an
						// intermediate consumer, we at least have a defined behavior, and the thus enforced ordering of tasks
						// likely reflects what the user expects.
						add_dependency(tsk.get(), &last_writer, dependency_kind::ANTI_DEP);
					}
				}

				buffers_last_writers.at(bid).update_region(write_requirements, tid);
			}
		}

		if(auto cgid = tsk->get_collective_group_id(); cgid != 0) {
			if(auto prev = last_collective_tasks.find(cgid); prev != last_collective_tasks.end()) {
				add_dependency(tsk.get(), task_map.at(prev->second).get(), dependency_kind::ORDER_DEP);
				last_collective_tasks.erase(prev);
			}
			last_collective_tasks.emplace(cgid, tid);
		}

		if(last_barrier_task_id) {
			const auto deps = tsk->get_dependencies();
			if(std::find_if(deps.begin(), deps.end(), [](const auto& dep) { return dep.kind == dependency_kind::TRUE_DEP; }) == deps.end()) {
				add_dependency(tsk.get(), task_map.at(last_barrier_task_id).get(), dependency_kind::TRUE_DEP);
			}
		}

		if(tsk->get_type() == task_type::BARRIER) { last_barrier_task_id = tsk->get_id(); }
	}

	task& task_manager::register_task_internal(std::unique_ptr<task> task) {
		auto& task_ref = *task;
		assert(task != nullptr);
		task_map.emplace(task->get_id(), std::move(task));
		execution_front.insert(&task_ref);
		return task_ref;
	}

	void task_manager::invoke_callbacks(task_id tid, task_type type) {
		for(auto& cb : task_callbacks) {
			cb(tid, type);
		}
	}

	void task_manager::add_dependency(task* depender, task* dependee, dependency_kind kind) {
		assert(depender != dependee);
		assert(depender != nullptr && dependee != nullptr);
		depender->add_dependency({dependee, kind});
		execution_front.erase(dependee);
		max_pseudo_critical_path_length = std::max(max_pseudo_critical_path_length, depender->get_pseudo_critical_path_length());
	}

	void task_manager::apply_milestone(task* new_milestone, const task* new_task_horizon) {
		// precondition: caller holds a lock on task_mutex

		// add dependencies from a copy of the front to this task
		auto current_front = get_execution_front();
		for(task* front_task : current_front) {
			if(front_task != new_milestone) { add_dependency(new_milestone, front_task); }
		}

		// apply the previous horizon to buffers_last_writers and last_collective_tasks data structs
		if(new_task_horizon != nullptr) {
			const task_id prev_cptid = new_task_horizon->get_id();
			for(auto& [_, buffer_region_map] : buffers_last_writers) {
				buffer_region_map.apply_to_values([prev_cptid](std::optional<task_id> tid) -> std::optional<task_id> {
					if(!tid) return tid;
					return {std::max(prev_cptid, *tid)};
				});
			}
			for(auto& [cgid, tid] : last_collective_tasks) {
				tid = std::max(prev_cptid, tid);
			}

			// We also use the previous horizon as the new init task for host-initialized buffers
			current_init_task_id = prev_cptid;
		}
	}

	void task_manager::generate_task_horizon() {
		// we are probably overzealous in locking here
		{
			std::lock_guard lock(task_mutex);
			current_milestone_critical_path_length = max_pseudo_critical_path_length;
			const auto new_horizon_task = &register_task_internal(task::make_horizon_task(get_new_tid()));
			apply_milestone(new_horizon_task, current_milestone_task);
			current_milestone_task = new_horizon_task;
		}

		// it's important that we don't hold the lock while doing this
		invoke_callbacks(current_milestone_task->get_id(), task_type::HORIZON);
	}

	task_id task_manager::create_barrier_task() {
		// we are probably overzealous in locking here
		{
			std::lock_guard lock(task_mutex);
			current_milestone_task = &register_task_internal(task::make_barrier(get_new_tid()));
			apply_milestone(current_milestone_task, current_milestone_task);
			compute_dependencies(current_milestone_task->get_id());
			current_milestone_critical_path_length = max_pseudo_critical_path_length;
		}

		// it's important that we don't hold the lock while doing this
		invoke_callbacks(current_milestone_task->get_id(), task_type::BARRIER);
		return current_milestone_task->get_id();
	}

	void task_manager::clean_up_pre_milestone_tasks() {
		task_id deletion_task_id = milestone_task_id_for_deletion.exchange(nothing_to_delete);
		if(deletion_task_id != nothing_to_delete) {
			for(auto iter = task_map.begin(); iter != task_map.end();) {
				if(iter->first < deletion_task_id) {
					iter = task_map.erase(iter);
				} else {
					++iter;
				}
			}
		}
	}

	void task_manager::wait_on_barrier(task_id tid) {
#ifndef NDEBUG
		{
			std::lock_guard lock(task_mutex);
			assert(task_map.at(tid)->get_type() == task_type::BARRIER);
		}
#endif

		std::unique_lock lock{barrier_mutex};
		barrier_executed.wait(lock, [=] { return last_executed_barrier >= tid; });
	}

} // namespace detail
} // namespace celerity
