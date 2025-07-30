/*
 * EMAlgorithm.cpp
 *
 *  Created on: Mar 8, 2017
 *      Author: wkwedlo
 */

#include <cmath>
#include <limits>

#include "EMAlgorithm.h"
#include "EMException.h"
#include "CachedEM.h"
#include "MatrixEM.h"

#include "../Util/OpenMP.h"
#include "../Util/StdOut.h"
#include "../Util/NumaAlloc.h"
#include "../Util/Rand.h"
#include "../Util/PrecisionTimer.h"


//#define TRACE_PLL

void EMAlgorithm::SetParameters(int aMaxSweeps,double aEps,double aAbortThr,double aRegCoeff) {
	MaxSweeps=aMaxSweeps;
	Eps=aEps;
	AbortThr=aAbortThr;
    RegCoef=aRegCoeff;
}

EMAlgorithm::EMAlgorithm(int ncols,long nrows,int ncl) {
	K=ncl;
	nCols=ncols;
	IterCounter=0;


    _vt_classdef("EMAlgorithm",&classID);
    _vt_funcdef("Loglikelihood",classID,&regionLogl);

	MaxSweeps=1000000;
	Eps=1e-5;
	AbortThr=100000.0;
    RegCoef=0.0;


    Posteriors.SetSize(nrows,K,1,false);
    Dataset.SetSize(nrows,nCols,1,false);
	Covs.SetSize(K);
	Means.resize(K,nCols);
	Densities.SetSize(K);
	Probs.resize(K);
	PLLs.SetSize(K);
	for(int i=0;i<K;i++) {
		Covs[i].resize(nCols,nCols);
		Densities[i]=new NormalDensity(nCols);
	}
	OMPData.SetSize(omp_get_max_threads());
#pragma omp parallel
	{
		int tid=omp_get_thread_num();
		OMPData[tid].arrF.SetSize(K);
		OMPData[tid].PLLs.SetSize(K);
	}

}


EMAlgorithm::~EMAlgorithm() {
	for(int i=0;i<K;i++)
		delete Densities[i];
}


void EMAlgorithm::InitDensities() {
	int ErrorCount=0;
#pragma omp parallel for reduction(+:ErrorCount) OMPDYNAMIC
	for(int i=0;i<K;i++) {
		try {
			Densities[i]->Init(Covs[i],Means.row(i));
		} catch (EMException &E) {
			ErrorCount++;
		}
	}
	if (ErrorCount>0)
		throw EMException("NormalDensity::Init failed inside a parallel region");
}

void EMAlgorithm::ComputePLLs() {
	int nThreads=omp_get_max_threads();
	for(int j=0;j<K;j++) PLLs[j]=0.0;
	for(int i=0;i<nThreads;i++) {
		for(int j=0;j<K;j++)
			PLLs[j]+=OMPData[i].PLLs[j];
	}
#ifdef TRACE_PLL
	OPTFLOAT PLLSum=0.0;
	for(int j=0;j<K;j++) {
		TRACE1("%g ",PLLs[j]);
		PLLSum+=PLLs[j];
	}
	TRACE1("Sum: %g\n",PLLSum);
#endif
}


void EMAlgorithm::RetrieveClasses(LargeVector<int> &Classes) const {
	const long nRows=Posteriors.GetRowCount();
#pragma omp parallel for firstprivate(nRows)
	for(long i=0;i<nRows;i++) {
		OPTFLOAT maxP=-1.0;
		OPTFLOAT maxj=-1;
		for(int j=0;j<K;j++) {
			if (Posteriors(i,j)>maxP) {
				maxj=j;
				maxP=Posteriors(i,j);
			}
			ASSERT(maxj>=0);
			Classes[i]=maxj;
		}
	}
}


void EMAlgorithm::DecodeMixture(const GaussianMixture &G) {
	for (int i=0;i<K;i++) {
		Probs[i]=G.GetWeight(i);
		Means.row(i)=G.GetMean(i);
		Covs[i]=G.GetCovariance(i);
	}
}

OPTFLOAT EMAlgorithm::Train(const MultithreadedDataset &Data,GaussianMixture &G,int Verbosity,bool SVD){
    CopyDataset(Data);
    TotalRowCount=Data.GetTotalRowCount();
    return TrainNoCopy(G,Verbosity,SVD);
}

OPTFLOAT EMAlgorithm::TrainNoCopy(GaussianMixture &G, int Verbosity, bool SVD)
 {
	PrecisionTimer T(CLOCK_MONOTONIC);
	DecodeMixture(G);
	OPTFLOAT PrevLogL=-std::numeric_limits<OPTFLOAT>::max(),LogL=PrevLogL;
	OPTFLOAT MaxCond=0.0;
    RegularizeCovariances();
	InitDensities();
	if (SVD) {
		MaxCond=FindLargestConditionNumber();
		if (MaxCond>AbortThr)
			throw EMException("Condition number of a covariance matrix is too high");
	}
	for(IterCounter=1;IterCounter<=MaxSweeps;IterCounter++) {
		PrecomputeDensityMatrix();
		LogL=ComputeLogLike();
		EStep();
		MStep();
        RegularizeCovariances();
		InitDensities();
		if (SVD) {
			MaxCond=FindLargestConditionNumber();
			if (MaxCond>AbortThr)
				throw EMException("Condition number of a covariance matrix is too high");
		}
		//if (LogL<PrevLogL)
		//	throw EMException("Loglikelihood decreased in EM iteration");

		OPTFLOAT e=std::fabs((LogL-PrevLogL)/LogL);
		if (LogL<PrevLogL)
			e=0.0;
		if (SVD) {
			OPTFLOAT MinTrace=FindMinTrace();
			if (Verbosity>1)
				kma_printf("iter: %d LogL: %10.6f e: %g Cond: %g Trace: %g Time[ms]: %g\n",IterCounter,LogL,e,MaxCond,MinTrace,
						1000.0*T.GetTimeDiffAndReset());
		} else {
			if (Verbosity>1)
				kma_printf("iter: %d LogL: %10.6f e: %g Time[ms]: %g\n",IterCounter,LogL,e,1000.0*T.GetTimeDiffAndReset());
		}
		PrevLogL=LogL;
		if (e<Eps) {
			IterCounter++;
			break;
		}
	}
	
	DynamicArray<OPTFLOAT> Weights(K);
	for(int i=0;i<K;i++)
		Weights[i]=Probs[i];
	G.Init(Weights,Covs,Means);
	return LogL;
}

OPTFLOAT EMAlgorithm::FindLargestConditionNumber() {
	OPTFLOAT MaxCond=0.0;
#pragma omp parallel for default(none) reduction(max:MaxCond) OMPDYNAMIC
	for(int i=0;i<K;i++) {
		OPTFLOAT Cond=Densities[i]->ComputeConditionNumber();
		if (Cond>MaxCond)
			MaxCond=Cond;
	}
	return MaxCond;
}
OPTFLOAT EMAlgorithm::FindMinTrace() {
	OPTFLOAT MinTrace=std::numeric_limits<OPTFLOAT>::max();
	for(int i=0;i<K;i++) {
		OPTFLOAT Trc=Densities[i]->ComputeTrace();
		if (Trc<MinTrace)
			MinTrace=Trc;
	}
	return MinTrace;
}

void EMAlgorithm::ComputePosterriorCorrelation(EigenMatrix &Corr) {
	long nRows=Posteriors.GetRowCount();
	int nCols=Posteriors.GetColCount();
	int nStride=Posteriors.GetStride();
	Eigen::Map<EigenMatrix> P(Posteriors.GetData(),nRows,nStride);
	Corr.noalias()=P.block(0,0,nRows,nCols).transpose()*P.block(0,0,nRows,nCols);
}

void EMAlgorithm::PrintNumaLocalityInfo() {
	kma_printf("NUMA locality of posterior matrix : %1.3f%%\n",100.0*NumaLocalityofLargeArray(Posteriors,1));
}

void EMAlgorithm::CopyDataset(const MultithreadedDataset &Data) {
    long nRows=Data.GetRowCount();
#pragma omp parallel for default(none) shared(Data) firstprivate(nRows,nCols)
    for(long i=0;i<nRows;i++) {
        const float  *row=Data.GetRowNew(i);
        OPTFLOAT *drow=Dataset(i);
#pragma omp simd
        for(int j=0;j<nCols;j++)
            drow[j]=row[j];
    }
}

void EMAlgorithm::RegularizeCovariances() {
    eigen_disable_malloc();
    for(int j=0;j<K;j++) {
        OPTFLOAT Trace=Covs[j].trace();
        Covs[j]=Covs[j]*(1.0-RegCoef)+RegCoef*Trace/(OPTFLOAT)nCols*EigenMatrix::Identity(nCols,nCols);
    }
    eigen_enable_malloc();
}


EMAlgorithm *CreateEMAlgorithm(const char *Name,const char *RName,const char *MName,int nCols,long nRows,long nTotalRows,int nCl,const char *MStepLoopParam) {
	if (Name==nullptr || !strcmp(Name,"matrixem") || !strcmp(Name,"simplematrixem") ) {
        {
            int Param1,Param2;
            BlockManagerBase::ParseEMBlockParams(MStepLoopParam,nRows,nCols,nCl,Param1,Param2);
            kma_printf("Blocks in density computation: %d, in Covariance computation: %d\n",Param1,Param2);
            if (Name==nullptr || !strcmp(Name,"matrixem")) {
                return new MatrixEM(nCols, nRows, nCl,RName, Param1,Param2);
            }
            else {
                return new SimpleMatrixEM(nCols, nRows, nCl, RName,Param1,Param2);
            }
        }
	}
	if (!strcmp(Name,"cachedem")) {
		int Param1,Param2;
		BlockManagerBase::ParseEMBlockParams(MStepLoopParam,nRows,nCols,nCl,Param1,Param2);
		kma_printf("Blocks in density computation: %d, in Covariance computation: %d\n",Param1,Param2);
        return new CachedEM(nCols,nRows,nCl,RName,Param1,Param2);
	}

	throw std::invalid_argument("Unknown EM algorithm");
}


