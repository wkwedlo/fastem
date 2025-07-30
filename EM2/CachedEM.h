#ifndef EM2_CACHEDEM_H_
#define EM2_CACHEDEM_H_

#include "EMAlgorithm.h"
#include "OpenMPEMReducer.h"


class CachedEM : public EMAlgorithm {

protected:
	EigenMatrix WeightedDensities;
	EigenVector WeightedDensSum;

	OMPReducer<Step1ReducerData> *pReducer1;
	OMPReducer<Step2ReducerData> *pReducer2;

    struct alignas(64) ThreadPrivateData {
    	EigenMatrix LogWeightDens;
    };
    DynamicArray<ThreadPrivateData> ThreadInfo;

	SimpleBlockManager BMDens;
    SimpleBlockManager BMMStep;


    OPTFLOAT LogLike;

	virtual void MStep();
	virtual void PrecomputeDensityMatrix();

    virtual void MPIReduceStep1Data() {}
    virtual void MPIReduceStep2Data() {}
	virtual void EStep() {}
	virtual OPTFLOAT ComputeLogLike() {return LogLike;}

    void MStepMeansPostSums();
    void MStepCovariances();

public:
	CachedEM(int ncols,long nrows,int ncl,const char *ReducerName,int anDensBlocks=1,int amStepBlocks=1); ;
	virtual ~CachedEM();
	virtual const char *GetName() const {return "CachedEM";}
	virtual void PrintNumaLocalityInfo(){}

    void ComputeMeansPostSum();
    void ComputeCovariances();
    

};
#endif