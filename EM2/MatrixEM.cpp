#include <cmath>
#include <unistd.h>
#include "../Util/ITAC.h"
#include "../Util/Profiler.h"
#include "../Util/OpenMP.h"
#include "MatrixEM.h"



MatrixEMBase::MatrixEMBase(int ncols, long nrows, int ncl, const char *ReducerName, int anDensBlocks,int anMStepBlocks) :
        EMAlgorithm(ncols,nrows,ncl) {

    Step1ReducerData x(ncl,nCols);
    Step2ReducerData y(ncl,nCols);

    if (ReducerName==nullptr || !strcmp(ReducerName,"naive")) {
        pReducer1=new NaiveOMPReducer<Step1ReducerData>(x);
        pReducer2=new NaiveOMPReducer<Step2ReducerData>(y);
    } else if (!strcmp(ReducerName,"log2")) {
        pReducer1=new Log2OMPReducer<Step1ReducerData>(x);
        pReducer2=new Log2OMPReducer<Step2ReducerData>(y);

    } else throw std::invalid_argument("Unknown OpenMP reducer name");

    PosteriorSum.resize(K);
    RowSum.resize(nCols);

    _vt_funcdef("PrecomputeDensityMatrix",classID,&regionDensities);

}

MatrixEMBase::~MatrixEMBase() {
    delete pReducer1;
    delete pReducer2;
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
    for (int j = 0; j < K; j++) {
        TempDataset.middleRows(0,Count)= MappedData.middleRows(DatasetOffset,Count).rowwise() - Means.row(j);
        cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, Count, nCols, 1.0, Densities[j]->GetInvertedCholCov().data(),
                    nCols, TempDataset.data(), nCols);
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
        EigenMatrix &tidCovSum = pReducer2->GetData(tid).CovSums[j];
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
    __FUNC_PROFILE_START;
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


void MatrixEM::doMStep(const LargeMatrix<OPTFLOAT> &Data,const LargeMatrix<OPTFLOAT> &Posteriors, DynamicArray<EigenMatrix> &Covs,
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
    pReducer2->ClearAllData();
#pragma omp parallel default(none) firstprivate(nRows) shared(MappedPosteriors,MappedData,Means)
    {
        int tid=omp_get_thread_num();
        auto &TempDataset=ThreadInfo[tid].TempDataset;
        long Offset,Count;

        const long nBlocks=BMMStep.GetBlockCount();
        for(int k=0;k<nBlocks;k++) {
            long StartRow,BlockRows;
            BMMStep.GetBlock(k,StartRow,BlockRows);
            BMMStep.GetThreadSubblock(StartRow,BlockRows,Offset,Count);
            BlockCovariances(Means, MappedData, MappedPosteriors, tid, TempDataset, StartRow+Offset, Count);
        }
    }
    //__EVENT_PROFILE_START("OpenMP Reducer2");
    _vt_end(regionMStepCovs);
    pReducer2->ReduceToFirstThread();
    MPIReduceStep2Data();
    //__EVENT_PROFILE_STOP("OpenMP Reducer2");
    for(int j=0;j<K;j++) {
        Covs[j]=pReducer2->GetData(0).CovSums[j];
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
    __FUNC_PROFILE_START;
    _vt_begin(regionDensities);
    long nRows=Dataset.GetRowCount();
    EigenRowVector LogDensCoeffs(K);

    CalcLogCoeffs(LogDensCoeffs);

    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedData(Dataset.GetData(),nRows,nCols,Eigen::OuterStride<>(Dataset.GetStride()));;
    Eigen::Map<EigenMatrix,0,Eigen::OuterStride<> > MappedPosteriors(Posteriors.GetData(),nRows,K,Eigen::OuterStride<>(Posteriors.GetStride()));


    eigen_disable_malloc();

    LogLike=0.0;
#pragma omp parallel  default (none) firstprivate(nRows) shared(LogDensCoeffs,MappedData,MappedPosteriors,stderr) reduction(+:LogLike)
    {
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
    }
    eigen_enable_malloc();
    _vt_end(regionDensities);
    MPIReduceLogL(LogLike);
    __FUNC_PROFILE_STOP;
}

void SimpleMatrixEM::doMStep(const LargeMatrix<OPTFLOAT> &Data,const LargeMatrix<OPTFLOAT> &Posteriors, DynamicArray<EigenMatrix> &Covs,
                        EigenMatrix &Means,EigenVector &Probs)
{
    __FUNC_PROFILE_START;
    long nRows=Posteriors.GetRowCount();
    eigen_disable_malloc();
    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedData(Data.GetData(),nRows,nCols,Eigen::OuterStride<>(Data.GetStride()));;
    Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > MappedPosteriors(Posteriors.GetData(),nRows,K,Eigen::OuterStride<>(Posteriors.GetStride()));
    __EVENT_PROFILE_START("SimpleMatrixEM::MStep means");
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
    pReducer1->ReduceToFirstThread();
    MPIReduceStep1Data();
    PosteriorSum=pReducer1->GetData(0).PosteriorSum;
    Means=pReducer1->GetData(0).MeanSums.array().colwise()/PosteriorSum.transpose().array();
    __EVENT_PROFILE_STOP("SimpleMatrixEM::MStep means");
    _vt_begin(regionMStepCovs);
    __EVENT_PROFILE_START("SimpleMatrixEM::MStep covariances");
    pReducer2->ClearAllData();
#pragma omp parallel default(none) firstprivate(nRows) shared(MappedPosteriors,MappedData,Means)
    {
        int tid=omp_get_thread_num();
        auto &TempDataset=ThreadInfo[tid].TempDataset;
        long Offset,Count;


        const long nBlocks=BMMStep.GetBlockCount();
        long PartitionStart,PartitionCount;
        BMMStep.GetThreadPartition(PartitionStart,PartitionCount);
        for(int k=0;k<nBlocks;k++) {
            BMMStep.GetThreadBlock(k, PartitionCount, Offset, Count);
            BlockCovariances(Means, MappedData, MappedPosteriors, tid, TempDataset, PartitionStart+Offset, Count);
        }
    }
    _vt_end(regionMStepCovs);
    __EVENT_PROFILE_START("OpenMP Reducer2");
    pReducer2->ReduceToFirstThread();
    __EVENT_PROFILE_STOP("OpenMP Reducer2");
    __EVENT_PROFILE_START("MPI Reducer2");
    MPIReduceStep2Data();
    __EVENT_PROFILE_STOP("MPI Reducer2");
    for(int j=0;j<K;j++) {
        Covs[j]=pReducer2->GetData(0).CovSums[j];
        Covs[j].template triangularView<Eigen::Lower>()=Covs[j].transpose();
    }
    __EVENT_PROFILE_STOP("SimpleMatrixEM::MStep covariances");
    Probs=PosteriorSum/(OPTFLOAT)TotalRowCount;
    eigen_enable_malloc();
    __FUNC_PROFILE_STOP;
}

