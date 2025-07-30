/*
 * ThreadAffinityInfo.cpp
 *
 *  Created on: Feb 19, 2018
 *      Author: wkwedlo
 */

#include <numa.h>
#include <sched.h>
#include "ThreadAffinityInfo.h"
#include "OpenMP.h"

ThreadAffinityInfo::ThreadAffinityInfo() {
	// TODO Auto-generated constructor stub
	NThreads=omp_get_max_threads();
	CPUs.SetSize(NThreads);
	Nodes.SetSize(NThreads);

}

void ThreadAffinityInfo::GetProcessInfo() {
#pragma omp parallel default(none)
	{
		int tid=omp_get_thread_num();
		CPUs[tid]=sched_getcpu();
		Nodes[tid]=numa_node_of_cpu(CPUs[tid]);
	}
}


void ThreadAffinityInfo::PrintReport(const char *fname) {
	GetProcessInfo();
	FILE *pF=fopen(fname,"w");
	if (pF!=NULL) {
		fprintf(pF,"Thread CPU placement info\n");
		PrintArray(pF,CPUs);
		fprintf(pF,"Thread NUMA node placement info\n");
		PrintArray(pF,Nodes);
		fclose(pF);
	}
}


ThreadAffinityInfo::~ThreadAffinityInfo() {
}

void ThreadAffinityInfo::PrintArray(FILE *pF,const DynamicArray<int> &Arr) {
	for(int j=0;j<NThreads;j++) {
		fprintf(pF,"%d",Arr[j]);
		if (j!=NThreads-1)
			fprintf(pF,",");
	}
	fprintf(pF,"\n");

}
