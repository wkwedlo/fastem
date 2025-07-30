//
// Created by wkwedlo on 24.04.23.
//
#include <unordered_map>
#include <chrono>

#include "StdOut.h"
#include "OpenMP.h"
#include "Profiler.h"

struct TimingInfo{
    int CallCount;
    double TotalTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> Start;
    std::chrono::time_point<std::chrono::high_resolution_clock> End;

    TimingInfo() {CallCount=0; TotalTime=0.0;}
};

std::unordered_map<std::string,TimingInfo> Map;

void __profile_start_timing(const std::string event) {
        if (omp_get_thread_num()>0 || omp_get_active_level()>1)
            return;
        TimingInfo &T=Map[event];
        T.CallCount++;
        T.Start=std::chrono::high_resolution_clock::now();
}

void __profile_stop_timing(const std::string event) {
    if (omp_get_thread_num()>0 || omp_get_active_level()>1)
        return;
    TimingInfo &T=Map[event];
    T.End=std::chrono::high_resolution_clock::now();
    double Duration = std::chrono::duration<double,std::milli>(T.End-T.Start).count();
    T.TotalTime+=Duration;
}

void __dump_profiles() {
    kma_printf("\n\nProfiling information\n");
    kma_printf("Function; Calls; Time/call [ms]; Total time [s]\n");
    for(const auto KeyVal: Map) {
        kma_printf("%s; %d; %g; %g\n",KeyVal.first.c_str(),KeyVal.second.CallCount,
                   KeyVal.second.TotalTime/KeyVal.second.CallCount,KeyVal.second.TotalTime/1000.0);

    }
}

