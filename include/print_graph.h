#pragma once

#include "task.h"

namespace celerity {
namespace detail {

	class command_graph;
	class logger;
	class task_manager;

	void print_task_graph(const std::unordered_map<task_id, std::unique_ptr<task>>& tdag, logger& graph_logger);
	void print_command_graph(const command_graph& cdag, logger& graph_logger, const task_manager& tm);

} // namespace detail
} // namespace celerity
