#include <cmath>
#include <unistd.h>
#ifndef EIGEN_USE_MKL_ALL
#include <cblas.h>
#endif

#include "../Util/ITAC.h"
#include "../Util/Profiler.h"
#include "../Util/OpenMP.h"
#include "../Util/Likwid.h"

#include "MatrixEM.h"
#include "EMException.h"


MatrixEMBase::MatrixEMBase(int ncols, long nrows, int ncl, const char *ReducerName, int anDensBlocks,int anMStepBlocks) :
        InvCholCovs(ncl,ncols), EMAlgorithm(ncols,nrows,ncl) {

    Step1ReducerData x(ncl,nCols);

    if (ReducerName==nullptr || !strcmp(ReducerName,"naive")) {
        pReducer1=new NaiveOMPReducer<Step1ReducerData>(x);
    } else if (!strcmp(ReducerName,"log2")) {
        pReducer1=new Log2OMPReducer<Step1ReducerData>(x);
    } else throw std::invalid_argument("Unknown OpenMP reducer name");

    PosteriorSum.resize(K);
    RowSum.resize(nCols);
    int nThreads=omp_get_max_threads();
    Step2CovSums.SetSize(nThreads);
#pragma omp parallel default(none) firstprivate(ncl,ncols)
    {
        int tid=omp_get_thread_num();
        Step2CovSums[tid].SetSize(ncl,ncols);
    }
    _vt_funcdef("PrecomputeDensityMatrix",classID,&regionDensities);

}

void MatrixEMBase::InitDensities() {
    int ErrorCount=0;
    __EVENT_PROFILE_START("Cholesky");
#pragma omp parallel for reduction(+:ErrorCount) OMPDYNAMIC
    for(int i=0;i<K;i++) {
        try {
            Densities[i]->Init(Covs[i],Means.row(i),InvCholCovs[i]);
        } catch (EMException &E) {
            ErrorCount++;
        }
    }
    __EVENT_PROFILE_STOP("Cholesky");
    if (ErrorCount>0)
        throw EMException("NormalDensity::Init failed inside a parallel region");
}



MatrixEMBase::~MatrixEMBase() {
    delete pReducer1;
}



void MatrixEMBase::PostProcessDensityMatrix(long DatasetOffset, long Count,
                                            Eigen::Map<EigenMatrix,0,Eigen::OuterStride<> > &MappedPosteriors,
                                            EigenMatrix &LocalWeightDens, double &tidLogLike) {

    for (long i = 0; i < Count; i++) {
        auto row=LocalWeightDens.row(i);
        auto Posteriorrow=MappedPosteriors.row(DatasetOffset + i);
        OPTFLOAT MaxLog=row.maxCoeff();
        OPTFLOAT LogSumExp=std::log((row.array()-MaxLog).exp().sum())+MaxLog;
        tidLogLike+=LogSumExp;
        Posteriorrow = (row.array()-LogSumExp).exp();
    }
}

void MatrixEMBase::ComputeWeightedTempDataset(int j, long DatasetOffset, long Count, Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > &MappedData, Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > &MappedPosteriors, const EigenMatrix &Means) {
    long nRows=MappedData.rows();
    EigenMatrix &TempDataset=ThreadInfo[omp_get_thread_num()].TempDataset;
    const OPTFLOAT * __restrict__ Mean=Means.row(j).data();
    OPTFLOAT InvSqrtPostSum=1.0/std::sqrt(PosteriorSum[j]);
    for(long i=0;i<Count;i++) {
        OPTFLOAT SqrtPost=std::sqrt(MappedPosteriors(DatasetOffset+i,j))*InvSqrtPostSum;
        OPTFLOAT * __restrict__ DstRow=TempDataset.row(i).data();
        const OPTFLOAT * __restrict__ SrcRow=MappedData.row(DatasetOffset+i).data();
#pragma omp simd
        for (int k=0;k<nCols;k++) {
            DstRow[k]=((OPTFLOAT)SrcRow[k]-Mean[k])*SqrtPost;
        }
    }
}

void MatrixEMBase::BlockDensities(const EigenRowVector &LogDensCoeffs,
                                  Eigen::Map<const EigenMatrix, 0, Eigen::OuterStride<>> &MappedData,
                                  Eigen::Map<EigenMatrix, 0, Eigen::OuterStride<>> &MappedPosteriors,
                                  EigenMatrix &TempDataset, EigenMatrix &LocalWeightDens, long DatasetOffset,
                                  long Count, OPTFLOAT &LogLike) {
    int nStride=InvCholCovs.GetStride();
    for (int j = 0; j < K; j++) {
        TempDataset.middleRows(0,Count)= MappedData.middleRows(DatasetOffset,Count).rowwise() - Means.row(j);

        cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, Count, nCols, 1.0, InvCholCovs[j].data(),
                    nStride, TempDataset.data(), nCols);
        LocalWeightDens.col(j).middleRows(0,Count)=-0.5*TempDataset.middleRows(0,Count).array().square().rowwise().sum()+LogDensCoeffs[j];
    }
    PostProcessDensityMatrix(DatasetOffset, Count, MappedPosteriors, LocalWeightDens, LogLike);
}

void MatrixEMBase::CalcLogCoeffs(EigenRowVector &LogDensCoeffs) {
    OPTFLOAT LogCoeff= -0.5 * nCols * log(2.0 * M_PI);
    for(int j=0; j < K; j++)
        LogDensCoeffs[j]= log(Probs[j]) + LogCoeff - 0.5 * Densities[j]->GetLogDeterminant();
}


void MatrixEMBase::BlockCovariances(const EigenMatrix &Means,
                                    Eigen::Map<const EigenMatrix, 0, Eigen::OuterStride<>> &MappedData,
                                    Eigen::Map<const EigenMatrix, 0, Eigen::OuterStride<>> &MappedPosteriors, int tid,
                                    EigenMatrix &TempDataset, long DatasetOffset, long Count) {
    for (int j = 0; j < K; j++) {
        EigenAlignedMap &tidCovSum = Step2CovSums[tid][j];
        ComputeWeightedTempDataset(j, DatasetOffset, Count, MappedData, MappedPosteriors, Means);
        tidCovSum.template selfadjointView<Eigen::Upper>().rankUpdate(TempDataset.middleRows(0,Count).transpose());
    }
}


void MatrixEM::CopyDataset(const MultithreadedDataset &Data) {
    long nRows=Data.GetRowCount();
    int nCols=Data.GetColCount();
#pragma omp parallel default(none)  firstprivate(nRows,nCols) shared(Data)
    {
        long nBlocks=BMDens.GetBlockCount();
        for(int k=0; k < nBlocks; k++) {
            long StartRow,BlockRows,Offset,Count;
            BMDens.GetBlock(k, StartRow, BlockRows);
            BMDens.GetThreadSubblock(StartRow, BlockRows, Offset, Count);
            for(long i=StartRow+Offset;i<StartRow+Offset+Count;i++) {
                const float  *row=Data.GetRowNew(i);
                OPTFLOAT *drow=Dataset(i);
#pragma omp simd
                for(int j=0;j<nCols;j++)
                    drow[j]=row[j];
            }
        }
    }
}




void MatrixEM::PrecomputeDensityMatrix() {
    __EVENT_PROFILE_START("Estep");
    _vt_begin(regionDensities);
    long nRows=Dataset.GetRowCount();
    EigenRowVector LogDensCoeffs(K);
    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedData(Dataset.GetData(),nRows,nCols,Eigen::OuterStride<>(Dataset.GetStride()));;
    Eigen::Map<EigenMatrix,0,Eigen::OuterStride<> > MappedPosteriors(Posteriors.GetData(),nRows,K,Eigen::OuterStride<>(Posteriors.GetStride()));

    CalcLogCoeffs(LogDensCoeffs);



    eigen_disable_malloc();

    LogLike=0.0;
#pragma omp parallel  default (none) firstprivate(nRows) shared(LogDensCoeffs,MappedData,MappedPosteriors,stderr) reduction(+:LogLike)
    {
        long nBlocks=BMDens.GetBlockCount();
        int tid=omp_get_thread_num();
        for(int k=0; k < nBlocks; k++) {
            long StartRow,BlockRows,Offset,Count;
            BMDens.GetBlock(k, StartRow, BlockRows);
            BMDens.GetThreadSubblock(StartRow, BlockRows, Offset, Count);
            BlockDensities(LogDensCoeffs, MappedData, MappedPosteriors, ThreadInfo[tid].TempDataset,
                           ThreadInfo[tid].LocalWeightDens, StartRow + Offset, Count, LogLike);
        }
    }
    eigen_enable_malloc();
    _vt_end(regionDensities);
    MPIReduceLogL(LogLike);
    __FUNC_PROFILE_STOP;
}



MatrixEM::MatrixEM(int ncols, long nrows, int ncl,const char *ReducerName, int anDensBlocks,int anMStepBlocks) :
        MatrixEMBase(ncols,nrows,ncl,ReducerName,anDensBlocks,anMStepBlocks),BMDens(nrows,anDensBlocks), BMMStep(nrows,anMStepBlocks) {
    LogLike=0.0;
    ThreadInfo.SetSize(omp_get_max_threads());

#pragma omp parallel default(none) firstprivate(ncl,ncols)
    {
        int tid=omp_get_thread_num();
        long maxDensSubBlockSize=BMDens.GetThredMaxSubBlockSize();
        long maxMstepSubBlockSize=BMMStep.GetThredMaxSubBlockSize();
        ThreadInfo[tid].LocalWeightDens.resize(maxDensSubBlockSize,ncl);
        ThreadInfo[tid].TempDataset.resize(std::max(maxDensSubBlockSize,maxMstepSubBlockSize),ncols);
    }
}


void MatrixEM::doMStep(const LargeMatrix<OPTFLOAT> &Data,const LargeMatrix<OPTFLOAT> &Posteriors, CovMatrixArray &Covs,
                        EigenMatrix &Means,EigenVector &Probs)
{
    __FUNC_PROFILE_START;
    long nRows=Posteriors.GetRowCount();
    eigen_disable_malloc();
    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedData(Data.GetData(),nRows,nCols,Eigen::OuterStride<>(Data.GetStride()));;
    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedPosteriors(Posteriors.GetData(),nRows,K,Eigen::OuterStride<>(Posteriors.GetStride()));
    __EVENT_PROFILE_START("MatrixEM::MStep means");
    _vt_begin(regionMStepMeans);
#pragma omp parallel firstprivate(nRows) shared(MappedPosteriors,MappedData)
    {
        int tid=omp_get_thread_num();
        long Offset,Count;
        EigenRowVector &tidPostSum=pReducer1->GetData(tid).PosteriorSum;
        tidPostSum.setZero();
        EigenMatrix &tidMeanSums=pReducer1->GetData(tid).MeanSums;
        tidMeanSums.setZero();

        long nBlocks=BMMStep.GetBlockCount();
        for(int k=0;k<nBlocks;k++) {
            long StartRow,BlockRows;
            BMMStep.GetBlock(k,StartRow,BlockRows);
            BMMStep.GetThreadSubblock(StartRow,BlockRows,Offset,Count);
            tidPostSum+=MappedPosteriors.middleRows(StartRow+Offset,Count).colwise().sum();
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,K, nCols, Count, 1.0,MappedPosteriors.middleRows(
                    StartRow+Offset,Count).data(), K, MappedData.middleRows(StartRow+Offset,Count).data(), nCols, 1.0, tidMeanSums.data(), nCols);
        }
    }
    _vt_end(regionMStepMeans);
    pReducer1->ReduceToFirstThread();
    MPIReduceStep1Data();
    PosteriorSum=pReducer1->GetData(0).PosteriorSum;
    Means=pReducer1->GetData(0).MeanSums.array().colwise()/PosteriorSum.transpose().array();
    __EVENT_PROFILE_STOP("MatrixEM::MStep means");
    _vt_begin(regionMStepCovs);
    __EVENT_PROFILE_START("MatrixEM::MStep covariances");
#pragma omp parallel default(none) firstprivate(nRows) shared(MappedPosteriors,MappedData,Means)
    {
        int tid=omp_get_thread_num();
        auto &TempDataset=ThreadInfo[tid].TempDataset;
        long Offset,Count;
        Step2CovSums[tid].ClearData();
        const long nBlocks=BMMStep.GetBlockCount();
        for(int k=0;k<nBlocks;k++) {
            long StartRow,BlockRows;
            BMMStep.GetBlock(k,StartRow,BlockRows);
            BMMStep.GetThreadSubblock(StartRow,BlockRows,Offset,Count);
            BlockCovariances(Means, MappedData, MappedPosteriors, tid, TempDataset, StartRow+Offset, Count);
        }
        Step2CovSums[tid].PackInPlace();
    }
    //__EVENT_PROFILE_START("OpenMP Reducer2");
    _vt_end(regionMStepCovs);
    CovMatrixArray::SegmentedReduction(Step2CovSums);
    MPIReduceStep2Data();
   Step2CovSums[0].UnpackInPlace();
    //__EVENT_PROFILE_STOP("OpenMP Reducer2");
    for(int j=0;j<K;j++) {
        Covs[j]=Step2CovSums[0][j];
        Covs[j].template triangularView<Eigen::Lower>()=Covs[j].transpose();
    }
    __EVENT_PROFILE_STOP("MatrixEM::MStep covariances");
    Probs=PosteriorSum/(OPTFLOAT)TotalRowCount;
    eigen_enable_malloc();
    __FUNC_PROFILE_STOP;
}



void MatrixEM::MStep()
{
    doMStep(Dataset,Posteriors,Covs,Means,Probs);
}




void SimpleMatrixEM::MStep()
{
    doMStep(Dataset,Posteriors,Covs,Means,Probs);
}

SimpleMatrixEM::SimpleMatrixEM(int ncols, long nrows, int ncl,const char *ReducerName, int anDensBlocks,int anMStepBlocks) :
        MatrixEMBase(ncols,nrows,ncl,ReducerName,anDensBlocks,anMStepBlocks),BMDens(nrows,anDensBlocks), BMMStep(nrows,anMStepBlocks) {
    LogLike=0.0;
    ThreadInfo.SetSize(omp_get_max_threads());

#pragma omp parallel default(none) firstprivate(ncl,ncols)
    {
        int tid=omp_get_thread_num();
        long maxDensThreadBlockSize=BMDens.GetMaxThreadBlockSize();
        long maxMStepThreadBlockSize=BMMStep.GetMaxThreadBlockSize();
        ThreadInfo[tid].LocalWeightDens.resize(maxDensThreadBlockSize, ncl);
        ThreadInfo[tid].TempDataset.resize(std::max(maxDensThreadBlockSize,maxMStepThreadBlockSize), ncols);
    }

}

void SimpleMatrixEM::PrecomputeDensityMatrix() {
    __EVENT_PROFILE_START("EStep sequential");
    _vt_begin(regionDensities);
    long nRows=Dataset.GetRowCount();
    EigenRowVector LogDensCoeffs(K);

    CalcLogCoeffs(LogDensCoeffs);

    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedData(Dataset.GetData(),nRows,nCols,Eigen::OuterStride<>(Dataset.GetStride()));;
    Eigen::Map<EigenMatrix,0,Eigen::OuterStride<> > MappedPosteriors(Posteriors.GetData(),nRows,K,Eigen::OuterStride<>(Posteriors.GetStride()));
    eigen_disable_malloc();
    LogLike=0.0;
    __EVENT_PROFILE_STOP("EStep sequential");
    __EVENT_PROFILE_START("EStep");
#pragma omp parallel  default (none) firstprivate(nRows) shared(LogDensCoeffs,MappedData,MappedPosteriors,stderr) reduction(+:LogLike)
    {

        __likwid_markerStartRegion("EStep");
        long nBlocks=BMDens.GetBlockCount();
        int tid=omp_get_thread_num();
        long PartitionStart,PartitionCount;
        BMDens.GetThreadPartition(PartitionStart,PartitionCount);
        for(int k=0; k < nBlocks; k++) {
            long Offset,Count;
            BMDens.GetThreadBlock(k, PartitionCount, Offset, Count);
            BlockDensities(LogDensCoeffs, MappedData, MappedPosteriors, ThreadInfo[tid].TempDataset,
                           ThreadInfo[tid].LocalWeightDens, PartitionStart + Offset, Count, LogLike);
        }
        __likwid_markerStopRegion("EStep");
    }
    eigen_enable_malloc();
    _vt_end(regionDensities);
    __EVENT_PROFILE_STOP("EStep");
    __EVENT_PROFILE_START("Reductions MPI EStep");
    MPIReduceLogL(LogLike);
    __EVENT_PROFILE_STOP("Reductions MPI EStep");
}





void SimpleMatrixEM::FillCovsAndRegularize(CovMatrixArray &Covs) {

    __EVENT_PROFILE_START("MStep transpose and regularize");
    constexpr  int BLOCK_SIZE = 32;
    const int nStride = Covs.GetStride();
    const OPTFLOAT MinusRegCoef = 1.0 - RegCoef;
#pragma omp parallel for default(none) firstprivate(nStride,MinusRegCoef) shared(Covs)
    for (int k = 0; k < K; ++k) {
        OPTFLOAT RegCorrection=RegCoef*Step2CovSums[0][k].trace()/(OPTFLOAT)nCols;
        const OPTFLOAT* pSrc = Step2CovSums[0][k].data();
        OPTFLOAT * pDst = Covs[k].data();

        for (int i_blk = 0; i_blk < nCols; i_blk += BLOCK_SIZE) {
            for (int j_blk = i_blk; j_blk < nCols; j_blk += BLOCK_SIZE) {
                int i_end = std::min(i_blk + BLOCK_SIZE, nCols);
                int j_end = std::min(j_blk + BLOCK_SIZE, nCols);

                for (int i = i_blk; i < i_end; ++i) {
                    int j_start;
                    if (i_blk==j_blk) {
                        j_start=i+1;
                        OPTFLOAT val= pSrc[i * nStride+ i]*MinusRegCoef+RegCorrection;
                        pDst[i * nStride + i] = val;
                    } else {
                        j_start=j_blk;
                    }
                    for (int j = j_start; j < j_end; ++j) {
                        OPTFLOAT val = pSrc[i * nStride + j]*MinusRegCoef;
                        pDst[i * nStride + j] = val;
                        pDst[j * nStride + i] = val;
                    }
                }
            }
        }
    }
    __EVENT_PROFILE_STOP("MStep transpose and regularize");
}

void SimpleMatrixEM::doMStep(const LargeMatrix<OPTFLOAT> &Data,const LargeMatrix<OPTFLOAT> &Posteriors, CovMatrixArray &Covs,
                             EigenMatrix &Means,EigenVector &Probs)
{
    long nRows=Posteriors.GetRowCount();
    eigen_disable_malloc();
    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedData(Data.GetData(),nRows,nCols,Eigen::OuterStride<>(Data.GetStride()));;
    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedPosteriors(Posteriors.GetData(),nRows,K,Eigen::OuterStride<>(Posteriors.GetStride()));
    __EVENT_PROFILE_START("MStep means");
    _vt_begin(regionMStepMeans);
#pragma omp parallel firstprivate(nRows) shared(MappedPosteriors,MappedData)
    {
        int tid=omp_get_thread_num();
        long Offset,Count;
        EigenRowVector &tidPostSum=pReducer1->GetData(tid).PosteriorSum;
        tidPostSum.setZero();
        EigenMatrix &tidMeanSums=pReducer1->GetData(tid).MeanSums;
        EigenMatrix &tidTempDataset=ThreadInfo[tid].TempDataset;
        tidMeanSums.setZero();

        long nBlocks=BMMStep.GetBlockCount();
        long PartitionStart,PartitionCount;
        BMMStep.GetThreadPartition(PartitionStart,PartitionCount);
        for(int k=0;k<nBlocks;k++) {
            BMMStep.GetThreadBlock(k, PartitionCount, Offset, Count);
            tidPostSum+=MappedPosteriors.middleRows(PartitionStart+Offset,Count).colwise().sum();
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,K, nCols, Count, 1.0,MappedPosteriors.middleRows(
                    PartitionStart+Offset,Count).data(), K, MappedData.middleRows(PartitionStart+Offset,Count).data(), nCols, 1.0, tidMeanSums.data(), nCols);
        }
    }
    _vt_end(regionMStepMeans);
    __EVENT_PROFILE_STOP("MStep means");
    __EVENT_PROFILE_START("Reductions OpenMP means");
    pReducer1->ReduceToFirstThread();
    __EVENT_PROFILE_STOP("Reductions OpenMP means");
    __EVENT_PROFILE_START("Reductions MPI means");
    MPIReduceStep1Data();
    __EVENT_PROFILE_STOP("Reductions MPI means");
    PosteriorSum=pReducer1->GetData(0).PosteriorSum;
    Means=pReducer1->GetData(0).MeanSums.array().colwise()/PosteriorSum.transpose().array();
    _vt_begin(regionMStepCovs);
    __EVENT_PROFILE_START("MStep covariances");
#pragma omp parallel default(none) firstprivate(nRows) shared(MappedPosteriors,MappedData,Means)
    {
        __likwid_markerStartRegion("MStep");
        int tid=omp_get_thread_num();
        auto &TempDataset=ThreadInfo[tid].TempDataset;
        long Offset,Count;


        const long nBlocks=BMMStep.GetBlockCount();
        long PartitionStart,PartitionCount;
        BMMStep.GetThreadPartition(PartitionStart,PartitionCount);
        Step2CovSums[tid].ClearData();
        for(int k=0;k<nBlocks;k++) {
            BMMStep.GetThreadBlock(k, PartitionCount, Offset, Count);
            BlockCovariances(Means, MappedData, MappedPosteriors, tid, TempDataset, PartitionStart+Offset, Count);
        }
        Step2CovSums[tid].PackInPlace();
        __likwid_markerStopRegion("MStep");
    }
    _vt_end(regionMStepCovs);
    __EVENT_PROFILE_STOP("MStep covariances");
    __EVENT_PROFILE_START("Reductions OpenMP covariances");
    CovMatrixArray::SegmentedReduction(Step2CovSums);
    __EVENT_PROFILE_STOP("Reductions OpenMP covariances");
    __EVENT_PROFILE_START("Reductions MPI covariances");
    MPIReduceStep2Data();
    Step2CovSums[0].UnpackInPlace();
    __EVENT_PROFILE_STOP("Reductions MPI covariances");
    FillCovsAndRegularize(Covs);
    Probs=PosteriorSum/(OPTFLOAT)TotalRowCount;
    eigen_enable_malloc();
}

