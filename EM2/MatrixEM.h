#ifndef EM2_MATRIXEM_H_
#define EM2_MATRIXEM_H_

#include "CachedEM.h"
#include "BlockManager.h"



class MatrixEMBase : public EMAlgorithm {
protected:
    OMPReducer<Step1ReducerData> *pReducer1;
    OMPReducer<Step2ReducerData> *pReducer2;


    struct alignas(64) ThreadPrivateData {
        EigenMatrix TempDataset;
        EigenMatrix LocalWeightDens;
        EigenVector WeightedDensSum;
    };

    DynamicArray<ThreadPrivateData> ThreadInfo;

    EigenRowVector PosteriorSum;
    EigenRowVector RowSum;


    virtual void MPIReduceStep1Data() {}
    virtual void MPIReduceStep2Data() {}

    /// ITAC regions
    int regionMStepMeans,regionMStepCovs,classID;


    void BlockDensities(const EigenRowVector &LogDensCoeffs,
                        Eigen::Map<const EigenMatrix, 0, Eigen::OuterStride<>> &MappedData,
                        Eigen::Map<EigenMatrix, 0, Eigen::OuterStride<>> &MappedPosteriors, EigenMatrix &TempDataset,
                        EigenMatrix &LocalWeightDens, long DatasetOffset, long Count, OPTFLOAT &LogLike);


    void BlockCovariances(const EigenMatrix &Means, Eigen::Map<const EigenMatrix, 0, Eigen::OuterStride<>> &MappedData,
                          Eigen::Map<const EigenMatrix, 0, Eigen::OuterStride<>> &MappedPosteriors, int tid,
                          EigenMatrix &TempDataset, long DatasetOffset, long Count);

    void ComputeWeightedTempDataset(int j, long DatasetOffset, long Count, Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > &MappedData, Eigen::Map<const EigenMatrix,0,Eigen::OuterStride<> > &MappedPosteriors, const EigenMatrix &Means);

    void CalcLogCoeffs(EigenRowVector &LogDensCoeffs);



    void PostProcessDensityMatrix(long DatasetOffset, long Count,
                                  Eigen::Map<EigenMatrix,0,Eigen::OuterStride<> > &MappedPosteriors,
                                  EigenMatrix &LocalWeightDens, double &tidLogLike);

    /// ITAC regions
    int regionDensities;
public:
    MatrixEMBase(int ncols,long nrows,int ncl,const char *ReducerName,int anDensBlocks=1, int anMStepBlocks=1);
    virtual ~MatrixEMBase();
};



class MatrixEM : public MatrixEMBase {
protected:
    BlockManager BMDens;
    BlockManager BMMStep;
    OPTFLOAT LogLike;
    virtual OPTFLOAT ComputeLogLike() {return LogLike;}
    virtual void PrecomputeDensityMatrix();
    virtual void EStep() {}
    virtual void MStep();
    virtual void CopyDataset(const MultithreadedDataset &Data);
    void doMStep(const LargeMatrix<OPTFLOAT> &Data,const LargeMatrix<OPTFLOAT> &Posteriors, DynamicArray<EigenMatrix> &Covs,
               EigenMatrix &Means,EigenVector &Probs);


public:
    MatrixEM(int ncols, long nrows, int ncl, const char *ReducerName,int anDensBlocks=1,int anMStepBlocks=1);
    virtual const char *GetName() const {return "MatrixEM";}
    virtual ~MatrixEM() {}
};

class SimpleMatrixEM : public MatrixEMBase {
protected:
    SimpleBlockManager BMDens;
    SimpleBlockManager BMMStep;
    OPTFLOAT LogLike;
    virtual OPTFLOAT ComputeLogLike() {return LogLike;}
    virtual void PrecomputeDensityMatrix();
    virtual void EStep() {}
    virtual void MStep();
    void doMStep(const LargeMatrix<OPTFLOAT> &Data,const LargeMatrix<OPTFLOAT> &Posteriors, DynamicArray<EigenMatrix> &Covs,
                 EigenMatrix &Means,EigenVector &Probs);


public:
    SimpleMatrixEM(int ncols, long nrows, int ncl, const char *ReducerName,int anDensBlocks=1,int anMStepBlocks=1);
    virtual const char *GetName() const {return "SimpleMatrixEM";}
    virtual ~SimpleMatrixEM() {}
};

#endif