//
// Created by wkwedlo on 24.04.23.
//

#ifndef CLUST_PROFILER_H
#define CLUST_PROFILER_H
#include <string>


#define PROFILER

void __profile_start_timing(const std::string event);
void __profile_stop_timing(const std::string event);
void __dump_profiles();

#ifdef PROFILER
#define __FUNC_PROFILE_START __profile_start_timing(__PRETTY_FUNCTION__)
#define __FUNC_PROFILE_STOP __profile_stop_timing(__PRETTY_FUNCTION__)
#define __DUMP_PROFILES __dump_profiles()
#define __EVENT_PROFILE_START(x) __profile_start_timing(x)
#define __EVENT_PROFILE_STOP(x) __profile_stop_timing(x)
#else
#define __FUNC_PROFILE_START
#define __FUNC_PROFILE_STOP
#define __DUMP_PROFILES
#define __EVENT_PROFILE_START(x)
#define __EVENT_PROFILE_STOP(x)
#endif




#endif //CLUST_PROFILER_H
