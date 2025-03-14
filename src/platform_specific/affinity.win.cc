#include <cassert>

#include <Windows.h>

#include "affinity.h"
#include "utils.h"

namespace celerity {
namespace detail {

	uint32_t affinity_cores_available() {
		using native_cpu_set = DWORD_PTR;

		native_cpu_set available_cores;
		[[maybe_unused]] native_cpu_set sys_affinity_mask;
		[[maybe_unused]] const auto ret = GetProcessAffinityMask(GetCurrentProcess(), &available_cores, &sys_affinity_mask);
		assert(ret != FALSE && "Error retrieving affinity mask.");
		return utils::popcount(available_cores);
	}

} // namespace detail
} // namespace celerity