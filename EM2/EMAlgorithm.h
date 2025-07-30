/*
 * EMAlgorithm.h
 *
 *  Created on: Mar 8, 2017
 *      Author: wkwedlo
 */

#ifndef EM2_EMALGORITHM_H_
#define EM2_EMALGORITHM_H_

#include "../Util/MultithreadedDataset.h"
#include "../Util/LargeMatrix.h"
#include "../Util/LargeVector.h"
#include "../Util/ITAC.h"

#include "NormalDensity.h"
#include "GaussianMixture.h"

struct ThreadData {
	DynamicArray <OPTFLOAT> arrF;
	DynamicArray <OPTFLOAT> PLLs;
};



class EMAlgorithm {
protected:
	int nCols;
	int K;
	int IterCounter;
    long TotalRowCount;

    int regionLogl,classID;

	int MaxSweeps;
	OPTFLOAT Eps;
	OPTFLOAT AbortThr;
    OPTFLOAT RegCoef;


    LargeMatrix<OPTFLOAT> Posteriors;
    LargeMatrix<OPTFLOAT> Dataset;


	DynamicArray<EigenMatrix> Covs;
	EigenMatrix Means;
	EigenVector Probs;
	DynamicArray<NormalDensity *>Densities;
	DynamicArray<OPTFLOAT> PLLs;
	DynamicArray <ThreadData> OMPData;

	virtual void MPIReduceLogL(OPTFLOAT &aLogL) {}

	OPTFLOAT FindLargestConditionNumber();
	OPTFLOAT FindMinTrace();
	void DecodeMixture(const GaussianMixture &G);
	virtual void EStep()=0;
	void ComputePLLs();
	virtual void InitDensities();
	virtual void PrecomputeDensityMatrix() {}
	virtual OPTFLOAT ComputeLogLike()=0;
	virtual void MStep()=0;
    virtual void CopyDataset(const MultithreadedDataset &Data);
    void RegularizeCovariances();
public:
	void ComputePosterriorCorrelation(EigenMatrix &Corr);
	void RetrieveClasses(LargeVector<int> &Classes) const;
	int GetIterCounter() const {return IterCounter;}
	void ClearIterCounter() {IterCounter=0;}
	int GetK() const {return K;}
	OPTFLOAT GetPLL(int k) const {return PLLs[k];}
	EMAlgorithm(int ncols,long nrows,int ncl);
	virtual ~EMAlgorithm();
	void SetParameters(int aMaxSweeps,double aEps,double aAbortThr,double aRegCoef=0.0);
	virtual OPTFLOAT Train(const MultithreadedDataset &Data,GaussianMixture &G,int Verbosity,bool SVD);
    virtual OPTFLOAT TrainNoCopy(GaussianMixture &G,int Verbosity,bool SVD);
	virtual void PrintNumaLocalityInfo();
	virtual const char *GetName() const=0;
};

EMAlgorithm *CreateEMAlgorithm(const char *Name,const char *RName,const char *MName,int nCols,long nRows,long nTotalRows,int nCl, const char *MStepLoopParam=nullptr);

#endif /* EM2_EMALGORITHM_H_ */
