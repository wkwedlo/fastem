#include "unistd.h"
#include "../Util/OpenMP.h"
#include "../Util/Profiler.h"
#include "CachedEM.h"


CachedEM::CachedEM(int ncols,long nrows,int ncl,const char *ReducerName,int anDensBlocks,int aMStepBlocks) :
            EMAlgorithm(ncols,nrows,ncl), BMDens(nrows,anDensBlocks), BMMStep(nrows,aMStepBlocks) {

    WeightedDensities.resize(nrows,ncl);
    WeightedDensSum.resize(nrows);

    Step1ReducerData x(ncl,nCols);
    Step2ReducerData y(ncl,nCols);

    if (ReducerName==nullptr || !strcmp(ReducerName,"naive")) {
        pReducer1=new NaiveOMPReducer<Step1ReducerData>(x);
        pReducer2=new NaiveOMPReducer<Step2ReducerData>(y);
    } else if (!strcmp(ReducerName,"log2")) {
        pReducer1=new Log2OMPReducer<Step1ReducerData>(x);
        pReducer2=new Log2OMPReducer<Step2ReducerData>(y);

    } else throw std::invalid_argument("Unknown OpenMP reducer name");

    LogLike=0.0;

    ThreadInfo.SetSize(omp_get_max_threads());

#pragma omp parallel default(none) firstprivate(ncl,ncols)
    {
        int tid=omp_get_thread_num();
        ThreadInfo[tid].LogWeightDens.resize(BMDens.GetMaxThreadBlockSize(),ncl);
    }
}

CachedEM::~CachedEM() {
    delete pReducer1;
    delete pReducer2;
}

void CachedEM::PrecomputeDensityMatrix() {
    __FUNC_PROFILE_START;
    long nRows = Dataset.GetRowCount();

    EigenRowVector LogProbs(K);
    for (int j = 0; j < K; j++)
        LogProbs[j] = std::log(Probs[j]);
    LogLike = 0.0;
    Eigen::Map<EigenMatrix, 0, Eigen::OuterStride<> > MappedPosteriors(Posteriors.GetData(), nRows, K,
                                                                       Eigen::OuterStride<>(Posteriors.GetStride()));

    eigen_disable_malloc();
#pragma  omp parallel default(none) shared(LogProbs, MappedPosteriors) firstprivate(nRows) reduction(+:LogLike)
    {
        DynamicArray<OPTFLOAT> &PLLs = OMPData[omp_get_thread_num()].PLLs;
        int tid=omp_get_thread_num();
        EigenMatrix &LogWeightDens = ThreadInfo[tid].LogWeightDens;

        long nBlocks=BMDens.GetBlockCount();
        long PartitionStart,PartitionCount;
        BMDens.GetThreadPartition(PartitionStart,PartitionCount);
        for(int k=0; k < nBlocks; k++) {
            long Offset,Count;
            BMDens.GetThreadBlock(k, PartitionCount, Offset, Count);
            long DatasetOffset=PartitionStart+Offset;

            for (int j=0;j<K;j++) {
                for (long i = 0; i < Count; i++)
                     LogWeightDens(i,j)= Densities[j]->GetLogDensity(Dataset(DatasetOffset+i)) + LogProbs[j];

            }
            for (long i=0;i<Count;i++) {
                auto row=LogWeightDens.row(i);
                OPTFLOAT MaxLog = row.maxCoeff();
                OPTFLOAT LogSumExp = std::log((row.array() - MaxLog).exp().sum()) + MaxLog;
                LogLike += LogSumExp;
                MappedPosteriors.row(DatasetOffset+i) = (row.array() - LogSumExp).exp();
            }
        }
    }
    eigen_enable_malloc();
    MPIReduceLogL(LogLike);
    __FUNC_PROFILE_STOP;
}

void CachedEM::ComputeMeansPostSum() {
    __FUNC_PROFILE_START;
    const long nRows=Dataset.GetRowCount();
    const int nCols=Dataset.GetColCount();

    eigen_disable_malloc();
#pragma omp parallel  default(none) firstprivate(nRows,nCols)
    {

        int tid=omp_get_thread_num();
        EigenRowVector &tidPostSum=pReducer1->GetData(tid).PosteriorSum;
        EigenMatrix &tidMeanSums=pReducer1->GetData(tid).MeanSums;
        tidPostSum=EigenRowVector::Zero(K);
        tidMeanSums=EigenMatrix::Zero(K,nCols);

#pragma omp for OMPDYNAMIC
        for(long i=0;i<nRows;i++) {
            const OPTFLOAT * __restrict row=Dataset(i);
            for(int k=0;k<K;k++) {
                const OPTFLOAT P=Posteriors(i,k);
                tidPostSum[k]+=P;
                for(int j=0;j<nCols;j++)
                    tidMeanSums(k,j)+=row[j]*P;
            }
        }
    }

    pReducer1->ReduceToFirstThread();
    MPIReduceStep1Data();
    Probs=pReducer1->GetData(0).PosteriorSum;
    Means=pReducer1->GetData(0).MeanSums.array().colwise()/Probs.array();
    eigen_enable_malloc();
    __FUNC_PROFILE_STOP;
}

void CachedEM::ComputeCovariances(){
    __FUNC_PROFILE_START;
    const long nRows=Dataset.GetRowCount();
    const int nCols=Dataset.GetColCount();
    eigen_disable_malloc();

#pragma omp parallel  default(none) firstprivate(nRows,nCols)
    {
        OPTFLOAT * __restrict__ work=(OPTFLOAT *)alloca(nCols*sizeof(OPTFLOAT));

        int tid=omp_get_thread_num();
        DynamicArray<EigenMatrix> &CovSums=pReducer2->GetData(tid).CovSums;
        for(int j=0;j<K;j++)
            CovSums[j]=EigenMatrix::Zero(nCols,nCols);


        long nBlocks=BMMStep.GetBlockCount();
        long PartitionStart,PartitionCount;
        BMMStep.GetThreadPartition(PartitionStart,PartitionCount);

        for(int k=0; k < nBlocks; k++) {
            long Offset, Count;
            BMMStep.GetThreadBlock(k, PartitionCount, Offset, Count);
            long DatasetOffset = PartitionStart + Offset;

            for (int j = 0; j < K; j++) {
                const OPTFLOAT *__restrict mean = Means.row(j).data();
                for (long i = 0; i < Count; i++) {
                    const OPTFLOAT *__restrict row = Dataset(DatasetOffset+i);
#pragma omp simd
                    for (int l = 0; l < nCols; l++)
                        work[l] = row[l] - mean[l];
                    blas_syr(CblasRowMajor, CblasUpper, nCols, Posteriors(DatasetOffset+i, j), work, 1, CovSums[j].data(), nCols);
                }
            }
        }
    }

    pReducer2->ReduceToFirstThread();
    MPIReduceStep2Data();

	for(int j=0;j<K;j++) {
			Covs[j]=(1.0/Probs[j])*pReducer2->GetData(0).CovSums[j].selfadjointView<Eigen::Upper>();
	}
    eigen_enable_malloc();
    __FUNC_PROFILE_STOP;
}


void CachedEM::MStep() {
    ComputeMeansPostSum();
    ComputeCovariances();
    Probs/=(OPTFLOAT)TotalRowCount;
}
