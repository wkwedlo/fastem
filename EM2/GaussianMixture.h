/*
 * GaussianMixture.h
 *
 *  Created on: Nov 12, 2012
 *      Author: wkwedlo
 */

#ifndef GAUSSIANMIXTURE_H_
#define GAUSSIANMIXTURE_H_


#include <stdio.h>

#include "NormalDensity.h"
#include "../Util/MultithreadedDataset.h"

class GaussianMixture {

	int nComponents;
	int nCols;
	DynamicArray <NormalDensity *> Densities;
	DynamicArray<OPTFLOAT> Weights;

	void AllocateData(int Comps);
	void FreeData();


public:
	void SplitComponent(int Idx,OPTFLOAT W1,const EigenRowVector &M1,const EigenMatrix &C1,OPTFLOAT W2,const EigenRowVector &M2,const EigenMatrix &C2);
	void MergeComponents(int Idx1,int Idx2,OPTFLOAT W,const EigenRowVector &M,const EigenMatrix &C);
	int GetComponentCount() const {return nComponents;}
	int GetDimension() const {return nCols;}
	OPTFLOAT GetWeight(int Component) const {return Weights[Component];}
	void SetWeight(int Component,OPTFLOAT Weight) {Weights[Component]=Weight;}
	const EigenMatrix GetCovariance(int Component) const {return Densities[Component]->GetCovariance();}
	const EigenRowVector GetMean(int Component) const {return Densities[Component]->GetMean();}
	void Init(const DynamicArray<OPTFLOAT> &aWeights,const DynamicArray<EigenMatrix> &aCovs,const EigenMatrix &aMeans);
	void InitComponent(int Component,OPTFLOAT Weight,const EigenMatrix &Cov,const EigenRowVector &Mean);
	void NormalizeWeights();

	OPTFLOAT Evaluate(const float *row) const;
	void ComputePosteriors(const float *row,EigenRowVector &Posterioros) const;
    int Classify(const float *row);
	void ClassifyData(const MultithreadedDataset &D,DynamicArray<int> &ClassNums);
	void WriteClasses(const char *fname,const MultithreadedDataset &D);
	OPTFLOAT FindLargestConditionNumber() const;
	void Dump() const;
	void Write(FILE *pF);
	void Write(const char *fname);

	GaussianMixture(const GaussianMixture &x);
	GaussianMixture& operator=(const GaussianMixture &x);

	OPTFLOAT ComponentBhattDist(BhattDist &Dist,int Comp, const GaussianMixture &G,int CG) const;
	GaussianMixture(int Comp,int Cols);
	GaussianMixture(FILE *pF);
	virtual ~GaussianMixture();
};

#endif /* GAUSSIANMIXTURE_H_ */
