#include <mpi.h>
#include <stdexcept>
#include <unistd.h>
#include <time.h>
#include <numa.h>

#ifdef INTEL_ITT
#include <ittnotify.h>
#endif

#include "../MPIKmeans/DistributedMultithreadedDataset.h"
#include "../MPIKmeans/MPIRank0StdOut.h"

#include "../Util/OpenMP.h"
#include "../Util/Compiler.h"
#include "../Util/Rand.h"
#include "../Util/PrecisionTimer.h"
#include "../Util/Profiler.h"
#include "../EM2/EMException.h"
#include "../EM2/EMArgs.h"
#include "../Util/MPIThreadAffinityInfo.h"


#include "MPIJainEMInitializer.h"
#include "MPICachedEM.h"
#include "MPIMatrixEM.h"



void InitRNGs() {
#ifdef _OPENMP
#pragma omp parallel
  	{
#pragma omp single
  		kma_printf("OpenMP version with %d threads\n",omp_get_num_threads());
  	}
#else
	kma_printf("Single threaded version\n");
#endif
	SRand(seed);
}

void DestroyRNGs() {
#pragma omp parallel
  	{
  		DelMTRand();
  	}
}

EMAlgorithm *CreateMPIEMAlgorithm(const char *Name,const char *RName,char *MName,const DistributedMultithreadedDataset &D,int nCl) {
	const int nCols=D.GetColCount();
    if (Name== nullptr ||  !strcmp(Name,"matrixem")) {
        int nRows=D.GetRowCount();
        int Param1,Param2;
        BlockManagerBase::ParseEMBlockParams(msteploopparam,nRows,nCols,nCl,Param1,Param2);
        kma_printf("Blocks in density computation: %d, in Covariance computation: %d\n",Param1,Param2);
        return new MPIMatrixEM(nCols, nRows, nCl,RName, Param1,Param2);
    }
    if (Name== nullptr ||  !strcmp(Name,"simplematrixem")) {
        int nRows=D.GetRowCount();
        int Param1,Param2;
        BlockManagerBase::ParseEMBlockParams(msteploopparam,nRows,nCols,nCl,Param1,Param2);
        kma_printf("Blocks in density computation: %d, in Covariance computation: %d\n",Param1,Param2);
        return new SimpleMPIMatrixEM(nCols, nRows, nCl,RName, Param1,Param2);
    }

    if (!strcmp(Name,"cachedem")) {
    	int nRows=D.GetRowCount();
    	int Param1,Param2;
    	BlockManagerBase::ParseEMBlockParams(msteploopparam,nRows,nCols,nCl,Param1,Param2);
    	kma_printf("Blocks in density computation: %d, in Covariance computation: %d\n",Param1,Param2);
        return new MPICachedEM(D.GetColCount(),D.GetRowCount(),nCl,RName,Param1,Param2);
    }
    throw std::invalid_argument("Unknown EM algorithm");
}



void PrintMPIandCPUInfo(int Rank) {
	char Name[MPI_MAX_PROCESSOR_NAME];
	int Len;
	MPI_Get_processor_name(Name,&Len);
#ifdef _OPENMP
	printf("For process %d  MPI_Get_processor_name returns %s\n",Rank,Name);
#pragma omp parallel
  	{
 		int CPU=sched_getcpu();
 		int Node=numa_node_of_cpu(CPU);
		printf("Process %d Thread %d CPU %d NUMA Node %d\n",Rank,omp_get_thread_num(),CPU,Node);
  	}
#else
  	{
		int CPU=sched_getcpu();
		int Node=numa_node_of_cpu(CPU);
		printf("For process %d  MPI_Get_processor_name returns %s, CPU %d NUMA Node %d\n",Rank,Name,CPU,Node);
  	}
#endif

}


int main(int argc, char *argv[])
{
#ifdef INTEL_ITT
    __itt_pause();
#endif
	int Rank,Size;
#ifdef _OPENMP
	int Provided;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&Provided);
#else
	MPI_Init(&argc,&argv);
#endif
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);
	MPIRank0StdOut::Init();

	try {
		ProcessArgs(argc,argv);
		if (verbosity>2)
			PrintMPIandCPUInfo(Rank);

		CompileHeader(argv[0]);
        NUMAAllocator::CreateNUMAAllocator(numaname,numaparam);
        NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
        pAlloc->PrintInfo();
        kma_printf("MPI application with %d processes\n",Size);
		InitRNGs();
		if (affname!=NULL) {
			MPIThreadAffinityInfo TAI;
			TAI.PrintReport(affname);
		}
		long nRows;
		int nCols;
		DistributedMultithreadedDataset::QueryHeader(fname,nRows,nCols);
		int nDensBlocks,nMStepBlocks;
		BlockManagerBase::ParseEMBlockParams(msteploopparam,nRows,nCols,ncl,nDensBlocks,nMStepBlocks);
		int nBlocks=std::max(nDensBlocks,nMStepBlocks);

		kma_printf("Loading dataset %s... ",fname);
		DistributedMultithreadedDataset D(nBlocks);
		PrecisionTimer T(CLOCK_MONOTONIC);
		D.Load(fname);
		MPI_Barrier(MPI_COMM_WORLD);
		double extime2=T.GetTimeDiff();

		kma_printf("%d vectors, %d features\n",D.GetTotalRowCount(),D.GetColCount());
		kma_printf("Loading done in %d blocks\n",nBlocks);
		kma_printf("Loading took %g seconds\n",extime2);
		kma_printf("Covariance matrix trace: %10.10g\n",D.GetCovMatrixTrace());
		kma_printf("Number of components: %d\n",ncl);
		kma_printf("Regularization factor: %g\n",regcoeff);
		kma_printf("EM convergence threshold: %g\n",Eps);
		kma_printf("EM abort threshold: %g\n",athr);

		GaussianMixture G(ncl,D.GetColCount());
		if (iname == nullptr) {
            MPIJainEMInitializer Init(D, ncl, fname);
            Init.Init(G);
        } else {
            FileEMInitializer Init(D,ncl,iname);
            Init.Init(G);
        }
		EMAlgorithm *pA=CreateMPIEMAlgorithm(aname,rname,mname,D,ncl);
		kma_printf("Using EM algorithm %s\n",pA->GetName());
		MPI_Barrier(MPI_COMM_WORLD);
        double LogL=0.0;
		try {
			if (burnin>0) {
				kma_printf("Running %d burn-in iterations\n",burnin);
				pA->SetParameters(burnin,Eps,athr,regcoeff);
				GaussianMixture G2=G;
				T.Reset();
				pA->Train(D,G2,verbosity,svd);
				double extime=T.GetTimeDiff();
				kma_printf("Burn-in time: %g seconds\n",extime);
				int IterCount=pA->GetIterCounter()-1;
				kma_printf("Burn-in iter time: %g miliseconds\n",1000.0*extime/(double)IterCount);
			}
			pA->SetParameters(maxiter,Eps,athr,regcoeff);
			T.Reset();
#ifdef INTEL_ITT
            __itt_resume();
#endif
            if (burnin>0) {
                LogL=pA->TrainNoCopy(G,verbosity,svd);
            } else {
                LogL=pA->Train(D,G,verbosity,svd);
            }
#ifdef INTEL_ITT
            __itt_pause();
#endif
			if (verbosity>3 && Rank==0)
				G.Dump();
		} catch (EMException &E) {
			kma_printf("MPIEM algorithm failed: %s\n",E.what());
		}
		MPI_Barrier(MPI_COMM_WORLD);
		extime2=T.GetTimeDiff();
		kma_printf("EM Execution time: %g seconds\n",extime2);
		int IterCount=pA->GetIterCounter()-1;
		kma_printf("%d total EM sweeps\n",IterCount);
		kma_printf("Iteration time: %g miliseconds\n",1000.0*extime2/(double)IterCount);
        kma_printf("Final loglikelihood: %g\n",LogL);
		__DUMP_PROFILES;
		DestroyRNGs();
	} catch (std::invalid_argument &E) {
		kma_printf("Invalid program argument: %s\n",E.what());
	}

	MPI_Finalize();
	StdOut::Destroy();
}
