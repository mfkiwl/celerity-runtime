#include "runtime.h"

#include <queue>
#include <string>
#include <unordered_map>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#include <mpi.h>

#include "affinity.h"
#include "buffer.h"
#include "buffer_manager.h"
#include "command_graph.h"
#include "executor.h"
#include "graph_generator.h"
#include "graph_serializer.h"
#include "host_object.h"
#include "log.h"
#include "mpi_support.h"
#include "scheduler.h"
#include "task_manager.h"
#include "user_bench.h"
#include "utils.h"
#include "version.h"

namespace celerity {
namespace detail {

	std::unique_ptr<runtime> runtime::instance = nullptr;

	void runtime::init(int* argc, char** argv[], cl::sycl::device* user_device) {
		if(test_mode) {
			instance.reset();
			instance = std::unique_ptr<runtime>(new runtime(argc, argv, user_device));
			return;
		}
		instance = std::unique_ptr<runtime>(new runtime(argc, argv, user_device));
	}

	runtime& runtime::get_instance() {
		if(instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
		return *instance;
	}

	static auto get_pid() {
#ifdef _MSC_VER
		return _getpid();
#else
		return getpid();
#endif
	}

	static std::string get_version_string() {
		using namespace celerity::version;
		return fmt::format("{}.{}.{} {}{}", major, minor, patch, git_revision, git_dirty ? "-dirty" : "");
	}

	static const char* get_build_type() {
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		return "debug";
#else
		return "release";
#endif
	}

	static std::string get_sycl_version() {
#if defined(__COMPUTECPP__)
		return fmt::format("ComputeCpp {}.{}.{}", COMPUTECPP_VERSION_MAJOR, COMPUTECPP_VERSION_MINOR, COMPUTECPP_VERSION_PATCH);
#elif defined(__HIPSYCL__) || defined(__HIPSYCL_TRANSFORM__)
		return fmt::format("hipSYCL {}.{}.{}", HIPSYCL_VERSION_MAJOR, HIPSYCL_VERSION_MINOR, HIPSYCL_VERSION_PATCH);
#elif CELERITY_DPCPP
		return "DPC++ / Clang " __clang_version__;
#else
#error "unknown SYCL implementation"
#endif
	}

	runtime::runtime(int* argc, char** argv[], cl::sycl::device* user_device) {
		if(!test_mode) {
			int provided;
			MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
			assert(provided == MPI_THREAD_MULTIPLE);
		}

		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		num_nodes = world_size;

		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		local_nid = world_rank;

		spdlog::set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e] [{:0{}}] [%^%l%$] %v", world_rank, int(ceil(log10(world_size)))));

		cfg = std::make_unique<config>(argc, argv);

		if(const uint32_t cores = affinity_cores_available(); cores < min_cores_needed) {
			CELERITY_WARN("Celerity has detected that only {} logical cores are available to this process. It is recommended to assign at least {} "
			              "logical cores. Performance may be negatively impacted.",
			    cores, min_cores_needed);
		}

		user_bench = std::make_unique<experimental::bench::detail::user_benchmarker>(*cfg, static_cast<node_id>(world_rank));

		h_queue = std::make_unique<host_queue>();
		d_queue = std::make_unique<device_queue>();

		// Initialize worker classes (but don't start them up yet)
		buffer_mngr = std::make_unique<buffer_manager>(*d_queue, [this](buffer_manager::buffer_lifecycle_event event, buffer_id bid) {
			switch(event) {
			case buffer_manager::buffer_lifecycle_event::REGISTERED: handle_buffer_registered(bid); break;
			case buffer_manager::buffer_lifecycle_event::UNREGISTERED: handle_buffer_unregistered(bid); break;
			default: assert(false && "Unexpected buffer lifecycle event");
			}
		});

		reduction_mngr = std::make_unique<reduction_manager>();
		host_object_mngr = std::make_unique<host_object_manager>();
		task_mngr = std::make_unique<task_manager>(num_nodes, h_queue.get(), reduction_mngr.get());
		exec = std::make_unique<executor>(local_nid, *h_queue, *d_queue, *task_mngr, *buffer_mngr, *reduction_mngr);
		if(is_master_node()) {
			cdag = std::make_unique<command_graph>();
			ggen = std::make_shared<graph_generator>(num_nodes, *task_mngr, *reduction_mngr, *cdag);
			gsrlzr = std::make_unique<graph_serializer>(*cdag,
			    [this](node_id target, const command_pkg& pkg, const std::vector<command_id>& dependencies) { flush_command(target, pkg, dependencies); });
			schdlr = std::make_unique<scheduler>(*ggen, *gsrlzr, num_nodes);
			task_mngr->register_task_callback([this](task_id tid, task_type type) { schdlr->notify_task_created(tid); });
		}

		CELERITY_INFO(
		    "Celerity runtime version {} running on {}. PID = {}, build type = {}", get_version_string(), get_sycl_version(), get_pid(), get_build_type());
		d_queue->init(*cfg, user_device);
	}

	runtime::~runtime() {
		if(is_master_node()) {
			schdlr.reset();
			gsrlzr.reset();
			ggen.reset();
			cdag.reset();
		}

		exec.reset();
		task_mngr.reset();
		reduction_mngr.reset();
		host_object_mngr.reset();
		// All buffers should have unregistered themselves by now.
		assert(!buffer_mngr->has_active_buffers());
		buffer_mngr.reset();
		d_queue.reset();
		h_queue.reset();
		user_bench.reset();

		// Make sure we free all of our MPI transfers before we finalize
		while(!active_flushes.empty()) {
			int done;
			MPI_Test(&active_flushes.begin()->req, &done, MPI_STATUS_IGNORE);
			if(done) { active_flushes.pop_front(); }
		}
		if(!test_mode) { MPI_Finalize(); }
	}

	void runtime::startup() {
		if(is_active) { throw runtime_already_started_error(); }
		is_active = true;
		if(is_master_node()) { schdlr->startup(); }
		exec->startup();
	}

	void runtime::shutdown() noexcept {
		assert(is_active);
		is_shutting_down = true;
		if(is_master_node()) {
			schdlr->shutdown();
			broadcast_control_command(command_type::SHUTDOWN, command_data{});
		}

		exec->shutdown();
		d_queue->wait();
		h_queue->wait();

		if(is_master_node() && cfg->get_log_level() == log_level::trace) {
			const auto print_max_nodes = cfg->get_graph_print_max_verts();
			{
				const auto graph_str = task_mngr->print_graph(print_max_nodes);
				if(graph_str.has_value()) {
					CELERITY_TRACE("Task graph:\n\n{}\n", *graph_str);
				} else {
					CELERITY_WARN("Task graph with {} vertices exceeds CELERITY_GRAPH_PRINT_MAX_VERTS={}. Skipping GraphViz output",
					    task_mngr->get_current_task_count(), print_max_nodes);
				}
			}
			{
				const auto graph_str = cdag->print_graph(print_max_nodes);
				if(graph_str.has_value()) {
					CELERITY_TRACE("Command graph:\n\n{}\n", *graph_str);
				} else {
					CELERITY_WARN("Command graph with {} vertices exceeds CELERITY_GRAPH_PRINT_MAX_VERTS={}. Skipping GraphViz output", cdag->command_count(),
					    print_max_nodes);
				}
			}
		}

		// Shutting down the task_manager will cause all buffers captured inside command group functions to unregister.
		// Since we check whether the runtime is still active upon unregistering, we have to set this to false first.
		is_active = false;
		task_mngr->shutdown();
		is_shutting_down = false;
		maybe_destroy_runtime();
	}

	void runtime::sync() noexcept {
		// (Note: currently this function busy waits, but sync is slow and shouldn't be on the critical path anyway)
		sync_id++;

		// First, broadcast SYNC command once the scheduler has finished all previous tasks
		if(is_master_node()) {
			while(!schdlr->is_idle()) {
				std::this_thread::yield();
			}
			sync_data cmd_data{};
			cmd_data.sync_id = sync_id;
			broadcast_control_command(command_type::SYNC, cmd_data);
		}

		// Then we wait for that sync to actually be reached.
		while(exec->get_highest_executed_sync_id() < sync_id) {
			std::this_thread::yield();
		}
	}

	task_manager& runtime::get_task_manager() const { return *task_mngr; }

	buffer_manager& runtime::get_buffer_manager() const { return *buffer_mngr; }

	reduction_manager& runtime::get_reduction_manager() const { return *reduction_mngr; }

	host_object_manager& runtime::get_host_object_manager() const { return *host_object_mngr; }

	void runtime::broadcast_control_command(command_type cmd, const command_data& data) {
		assert_true(is_master_node()) << "Control commands should only be broadcast from the master";
		for(auto n = 0u; n < num_nodes; ++n) {
			command_pkg pkg{next_control_command_id++, cmd, data};
			flush_command(n, pkg, {});
		}
	}

	void runtime::handle_buffer_registered(buffer_id bid) {
		const auto& info = buffer_mngr->get_buffer_info(bid);
		task_mngr->add_buffer(bid, info.range, info.is_host_initialized);
		if(is_master_node()) ggen->add_buffer(bid, info.range);
	}

	void runtime::handle_buffer_unregistered(buffer_id bid) { maybe_destroy_runtime(); }

	void runtime::maybe_destroy_runtime() const {
		if(is_active) return;
		if(is_shutting_down) return;
		if(buffer_mngr->has_active_buffers()) return;
		if(host_object_mngr->has_active_objects()) return;
		instance.reset();
	}

	void runtime::flush_command(node_id target, const command_pkg& pkg, const std::vector<command_id>& dependencies) {
		// Even though command packages are small enough to use a blocking send we want to be able to send to the master node as well,
		// which is why we have to use Isend after all. We also have to make sure that the buffer stays around until the send is complete.
		active_flushes.push_back(flush_handle{pkg, dependencies, MPI_REQUEST_NULL, {}});
		auto it = active_flushes.rbegin();
		it->data_type = mpi_support::build_single_use_composite_type(
		    {{sizeof(command_pkg), &it->pkg}, {sizeof(command_id) * dependencies.size(), it->dependencies.data()}});
		MPI_Isend(MPI_BOTTOM, 1, *it->data_type, static_cast<int>(target), mpi_support::TAG_CMD, MPI_COMM_WORLD, &active_flushes.rbegin()->req);

		// Cleanup finished transfers.
		// Just check the oldest flush. Since commands are small this will stay in equilibrium fairly quickly.
		int done;
		MPI_Test(&active_flushes.begin()->req, &done, MPI_STATUS_IGNORE);
		if(done) { active_flushes.pop_front(); }
	}

} // namespace detail
} // namespace celerity
