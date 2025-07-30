/*
 * EMInitializer.h
 *
 *  Created on: Apr 28, 2010
 *      Author: wkwedlo
 */

#ifndef EMINITIALIZER_H_
#define EMINITIALIZER_H_
#include <stdio.h>

#include "GaussianMixture.h"
#include "../Util/MultithreadedDataset.h"
#include "../Clust/KMAlgorithm.h"

class EMInitializer {

protected:
	int ncols;
	int Components;

	DynamicArray <EigenMatrix> Covs;
	EigenMatrix Means;
	DynamicArray<OPTFLOAT>    Weights;

	MultithreadedDataset &Data;


public:
	EMInitializer(MultithreadedDataset &D,int nC);
	virtual void Init(GaussianMixture &G)=0;
	virtual void PrintInfo()=0;
	virtual ~EMInitializer();
};


class JainEMInitializer : public EMInitializer {

public:
	JainEMInitializer(MultithreadedDataset &D,int nC);
	virtual void PrintInfo();
	virtual void Init(GaussianMixture &G);
	virtual ~JainEMInitializer();
};

class FileEMInitializer : public EMInitializer {
	FILE *pF=nullptr;
public:
	FileEMInitializer(MultithreadedDataset &D,int nC,const char *fname);
	virtual void PrintInfo();
	virtual void Init(GaussianMixture &G);
	virtual ~FileEMInitializer();
};

class KMeansEMInitializer : public EMInitializer {
protected:
	KMeansInitializer *pKMInit;
	NaiveKMA *pKMAlg;
	CentroidVector *pCV;
	CentroidRepair *pRepair;
	DynamicArray <OPTFLOAT> KMVec;
	DynamicArray<int> ClassNums;
	DynamicArray<int> ObjCounts;
public:
	KMeansEMInitializer(MultithreadedDataset &D,int nC);
	virtual void PrintInfo();
	virtual void Init(GaussianMixture &G);
	virtual ~KMeansEMInitializer();
};

class KMeansOrOrEMInitializer : public KMeansEMInitializer {
public:
	KMeansOrOrEMInitializer(MultithreadedDataset &D,int nC);
	virtual void PrintInfo();
	virtual ~KMeansOrOrEMInitializer() {}
};

EMInitializer *CreateEMInitializer(const char *name,MultithreadedDataset &Data,int nC);

#endif /* EMINITIALIZER_H_ */
