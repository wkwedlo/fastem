#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <stdexcept>

#ifdef INTEL_ITT
#include <ittnotify.h>
#endif

#include "../Util/OpenMP.h"
#include "../Util/Rand.h"
#include "../Util/StdOut.h"
#include "../Util/MultithreadedDataset.h"
#include "../Util/NumaAlloc.h"
#include "../Util/PrecisionTimer.h"
#include "../Util/Compiler.h"
#include "../Util/ThreadAffinityInfo.h"
#include "../Util/Profiler.h"

#include "EMArgs.h"
#include "MatrixEM.h"
#include "CachedEM.h"
#include "GaussianMixture.h"
#include "EMInitializer.h"
#include "EMException.h"
#include "OpenMPEMReducer.h"


void InitRNGs() {
  	printf("Seeed of rng: %d\n",seed);
#ifdef _OPENMP
#pragma omp parallel default(none)
  	{
#pragma omp single
  		printf("OpenMP version with %d threads\n",omp_get_num_threads());
  	}
#else
	printf("Single threaded version\n");
#endif
	SRand(seed);
}

void DestroyRNGs() {
#pragma omp parallel default(none)
  	{
  		DelMTRand();
  	}
}




void RunEM() {

	long nRows;
	int nCols;
	MultithreadedDataset::QueryHeader(fname,nRows,nCols);
	int nDensBlocks,nMStepBlocks;
	BlockManagerBase::ParseEMBlockParams(msteploopparam,nRows,nCols,ncl,nDensBlocks,nMStepBlocks);
	int nBlocks=std::max(nDensBlocks,nMStepBlocks);
	MultithreadedDataset D(nBlocks);

	kma_printf("Loading dataset %s... ",fname);
	PrecisionTimer T(CLOCK_MONOTONIC);
	D.Load(fname);
  	double extime=T.GetTimeDiff();
	kma_printf("%ld vectors, %d features\n",D.GetRowCount(),D.GetColCount());
	kma_printf("Loading done in %d blocks\n",nBlocks);
	kma_printf("Loading took %g seconds\n",extime);
	kma_printf("Covariance matrix trace: %10.10g\n",D.GetCovMatrixTrace());
	kma_printf("Number of components: %d\n",ncl);
	kma_printf("Regularization factor: %g\n",regcoeff);
	kma_printf("EM convergence threshold: %g\n",Eps);
	kma_printf("EM abort threshold: %g\n",athr);
	GaussianMixture G(ncl,D.GetColCount());
	EMInitializer *pInit=CreateEMInitializer(iname,D,ncl);
	JainEMInitializer Init(D,ncl);
	pInit->PrintInfo();
	pInit->Init(G);
	EMAlgorithm *pA=CreateEMAlgorithm(aname,rname,mname,D.GetColCount(),D.GetRowCount(),D.GetRowCount(),ncl,msteploopparam);
	kma_printf("Using EM algorithm %s\n",pA->GetName());
	double LogL;
	try {
		if (burnin>0) {
			kma_printf("Running %d burn-in iterations\n",burnin);
			pA->SetParameters(burnin,Eps,athr,regcoeff);
			GaussianMixture G2=G;
			T.Reset();
			pA->Train(D,G2,verbosity,svd);
			extime=T.GetTimeDiff();
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

        if (verbosity>3)
			G.Dump();
	} catch (EMException &E) {
		kma_printf("EM algorithm failed: %s\n",E.what());
	}
  	extime=T.GetTimeDiff();
	kma_printf("EM Execution time: %g seconds\n",extime);
	int IterCount=pA->GetIterCounter()-1;
	kma_printf("%d total EM sweeps\n",IterCount);
	kma_printf("Iteration time: %g miliseconds\n",1000.0*extime/(double)IterCount);
	kma_printf("Final loglikelihood: %g\n",LogL);
	if (verbosity>0) {
        pA->PrintNumaLocalityInfo();
    }
	delete pInit;
	delete pA;
}




int main(int argc, char *argv[])
{
#ifdef INTEL_ITT
    __itt_pause();
#endif
    StandardStdOut::Init();
	Eigen::initParallel();
	char hostname[1024];
	gethostname(hostname,1023);
	kma_printf("Running on machine: %s\n",hostname);
	try {
		ProcessArgs(argc,argv);
        NUMAAllocator::CreateNUMAAllocator(numaname,numaparam);
        NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
        pAlloc->PrintInfo();
		InitRNGs();
		CompileHeader(argv[0]);
		if (affname!=NULL) {
			ThreadAffinityInfo TAI;
			TAI.PrintReport(affname);
		}
		RunEM();
		DestroyRNGs();
        __DUMP_PROFILES;
	} catch (std::invalid_argument &E) {
		kma_printf("Invalid program argument: %s\n",E.what());
	}
	return 0;
}
