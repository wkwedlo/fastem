/*
 * CPUAffinityInfo.cpp
 *
 *  Created on: Feb 15, 2018
 *      Author: wkwedlo
 */

#include "MPIThreadAffinityInfo.h"

#include <mpi.h>

MPIThreadAffinityInfo::MPIThreadAffinityInfo() {
	MPI_Comm_size(MPI_COMM_WORLD,&MPISize);
	MPI_Comm_rank(MPI_COMM_WORLD,&MPIRank);
	if (MPIRank==0) {
		DstCPUs.SetSize(NThreads*MPISize);
		DstNodes.SetSize(NThreads*MPISize);
	}
}


void MPIThreadAffinityInfo::MergeMPIInfos() {
	MPI_Gather(CPUs.GetData(),NThreads,MPI_INT,DstCPUs.GetData(),NThreads,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Gather(Nodes.GetData(),NThreads,MPI_INT,DstNodes.GetData(),NThreads,MPI_INT,0,MPI_COMM_WORLD);
}

void MPIThreadAffinityInfo::PrintGlobalArray(FILE *pF,const DynamicArray<int> &DstArr) {
		for(int i=0;i<MPISize;i++) {
			for(int j=0;j<NThreads;j++) {
				fprintf(pF,"%d",DstArr[i*NThreads+j]);
				if (j!=NThreads-1)
					fprintf(pF,",");
			}
			fprintf(pF,"\n");
		}
}

void MPIThreadAffinityInfo::PrintReport(const char *fname) {
	GetProcessInfo();
	MergeMPIInfos();
	if (MPIRank==0) {
		FILE *pF=fopen(fname,"w");
		if (pF!=NULL) {
			fprintf(pF,"Thread CPU placement info\n");
			PrintGlobalArray(pF,DstCPUs);
			fprintf(pF,"Thread NUMA node placement info\n");
			PrintGlobalArray(pF,DstNodes);
			fclose(pF);
		}
	}
}

