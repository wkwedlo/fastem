/*
 * GaussianMixture.cpp
 *
 *  Created on: Nov 12, 2012
 *      Author: wkwedlo
 */

#include <time.h>
#include <math.h>
#include <unistd.h>
#include <iostream>

#include "GaussianMixture.h"

using std::cout;
using std::endl;


void GaussianMixture::Init(const DynamicArray<OPTFLOAT> &aWeights,const DynamicArray<EigenMatrix> &aCovs,const EigenMatrix &aMeans) {
	ASSERT(aWeights.GetSize()==nComponents);
	ASSERT(aCovs.GetSize()==nComponents);

	Weights=aWeights;
	for(int i=0;i<nComponents;i++)
		Densities[i]->Init(aCovs[i],aMeans.row(i));
}

void GaussianMixture::InitComponent(int Component,OPTFLOAT Weight,const EigenMatrix &Cov,const EigenRowVector &Mean) {
	Weights[Component]=Weight;
	Densities[Component]->Init(Cov,Mean);
}

void GaussianMixture::SplitComponent(int Idx,OPTFLOAT W1,const EigenRowVector &M1,const EigenMatrix &C1,OPTFLOAT W2,const EigenRowVector &M2,const EigenMatrix &C2) {
	Weights.Add(0.0);
	Densities.Add(nullptr);
	nComponents++;
	Densities[nComponents-1]=new NormalDensity(nCols);
	InitComponent(Idx,W1,C1,M1);
	InitComponent(nComponents-1,W2,C2,M2);
}

void GaussianMixture::MergeComponents(int Idx1,int Idx2,OPTFLOAT W,const EigenRowVector &M,const EigenMatrix &C) {
	ASSERT(Idx1!=Idx2);
	ASSERT(nComponents>1);
	
	InitComponent(Idx1,W,C,M);
	delete Densities[Idx2];
	Densities.RemoveAt(Idx2);
	Weights.RemoveAt(Idx2);
	nComponents--;
}


OPTFLOAT GaussianMixture::Evaluate(const float *row) const {
	OPTFLOAT Sum=0;
	for(int i=0;i<nComponents;i++) {
		Sum+=(Densities[i]->GetDensity(row)*(OPTFLOAT)Weights[i]);
	}
	return Sum;
}

void GaussianMixture::ComputePosteriors(const float *row,EigenRowVector &Posteriors) const {
	OPTFLOAT Sum=0;
	for(int i=0;i<nComponents;i++) {
		Posteriors[i]=Densities[i]->GetDensity(row)*(OPTFLOAT)Weights[i];
		Sum+=Posteriors[i];
	}
	Posteriors*=(1/Sum);
}

void GaussianMixture::NormalizeWeights() {
	OPTFLOAT Sum=0.0;
	for(int i=0;i<nComponents;i++) {
		Sum+=Weights[i];
	}
	for(int i=0;i<nComponents;i++) {
		Weights[i]*=(1/Sum);
	}
}

OPTFLOAT GaussianMixture::ComponentBhattDist(BhattDist &Dist,int Comp, const GaussianMixture &G,int CG) const {
	return Dist.ComputeDist(*Densities[Comp],*G.Densities[CG]);
}


int GaussianMixture::Classify(const float *row) {
	int Cls=0;
	OPTFLOAT MaxB=Densities[0]->GetDensity(row)*(OPTFLOAT)Weights[0];
	for(int j=1;j<nComponents;j++) {
		OPTFLOAT f=Densities[j]->GetDensity(row);
		OPTFLOAT B=Weights[j]*f;
		if (B>MaxB) {
			MaxB=B;
			Cls=j;
		}
	}
	return Cls;
}


void GaussianMixture::ClassifyData(const MultithreadedDataset &D,DynamicArray<int> &ClassNums)
{
	long nRows=D.GetRowCount();
	ClassNums.SetSize(nRows);
	for(long i=0;i<D.GetRowCount();i++) {
		const float *row=D.GetRowNew(i);
		ClassNums[i]=Classify(row);
	}
}


void GaussianMixture::WriteClasses(const char *fname,const MultithreadedDataset &D) {
	long nRows=D.GetRowCount();
	FILE *pF=fopen(fname,"wb");
	DynamicArray<int> ClassNums(nRows);
	ClassifyData(D,ClassNums);
	for(long i=0;i<nRows;i++)
		ClassNums[i]++;
	fwrite(&nRows,sizeof(int),1,pF);
	fwrite(ClassNums.GetData(),sizeof(int),nRows,pF);
	fclose(pF);
}


OPTFLOAT GaussianMixture::FindLargestConditionNumber() const
{
	OPTFLOAT maxcond=1.0;
	for(int i=0;i<nComponents;i++) {
		OPTFLOAT cond=Densities[i]->ComputeConditionNumber();
		if (cond>maxcond) {
			maxcond=cond;
		}
	}
	return maxcond;
}



void GaussianMixture::Dump() const{

	printf("Number of components: %d\n",nComponents);
	for(int i=0;i<nComponents;i++) {
		printf("Component %d: \n",i);
		EigenMatrix C=Densities[i]->GetCovariance();
		EigenRowVector M=Densities[i]->GetMean();
		printf("Mixing probability: %g\n",Weights[i]);
		printf("Mean vector: ");
		print(M);
		printf("Covariance matrix\n");
		print(C);
	}
	printf("Condition number of the solution: %g\n",FindLargestConditionNumber());
}

void GaussianMixture::Write(FILE *pF) {
	OPTFLOAT LastLogL=0.0;
	fwrite(&nComponents,sizeof(int),1,pF);
	fwrite(&nCols,sizeof(int),1,pF);
	for(int i=0;i<nComponents;i++) {
		fwrite((Densities[i]->GetCovariance()).data(),nCols*nCols,sizeof(OPTFLOAT),pF);
		fwrite((Densities[i]->GetMean()).data(),nCols,sizeof(OPTFLOAT),pF);
		fwrite(&(Weights[i]),sizeof(OPTFLOAT),1,pF);
	}
	fwrite(&LastLogL,sizeof(OPTFLOAT),1,pF);
}

void GaussianMixture::Write(const char *fname) {
	FILE *pF=fopen(fname,"wb");
	Write(pF);
	fclose(pF);
}

void GaussianMixture::AllocateData(int Comps) {
	Densities.SetSize(Comps);
	Weights.SetSize(Comps);
	for(int i=0;i<Comps;i++)
		Densities[i]=new NormalDensity(nCols);
}

void GaussianMixture::FreeData() {
	for(int i=0;i<nComponents;i++)
		delete Densities[i];

}

GaussianMixture::GaussianMixture(const GaussianMixture &x) {
	nCols=x.nCols;
	nComponents=x.nComponents;
	AllocateData(nComponents);
	for(int i=0;i<nComponents;i++) {
		EigenRowVector M=x.Densities[i]->GetMean();
		EigenMatrix C=x.Densities[i]->GetCovariance();
		Densities[i]->Init(C,M);
		Weights[i]=x.Weights[i];
	}
}
GaussianMixture& GaussianMixture::operator=(const GaussianMixture &x)
{
	ASSERT(nCols==x.nCols);
	

	if (nComponents!=x.nComponents) {
		FreeData();
		nComponents=x.nComponents;
		AllocateData(nComponents);
	}
	for(int i=0;i<nComponents;i++) {
		EigenRowVector M=x.Densities[i]->GetMean();
		EigenMatrix C=x.Densities[i]->GetCovariance();
		Densities[i]->Init(C,M);
		Weights[i]=x.Weights[i];
	}
	return *this;

}

GaussianMixture::GaussianMixture(int Comp,int Cols) {
	nComponents=Comp;
	nCols=Cols;
	AllocateData(nComponents);
}

GaussianMixture::~GaussianMixture() {
	FreeData();
}


GaussianMixture::GaussianMixture(FILE *pF) {
	auto nIntems=fread(&nComponents,sizeof(int),1,pF);
	nIntems=fread(&nCols,sizeof(int),1,pF);

	EigenMatrix C(nCols,nCols);
	EigenRowVector M(nCols);
	AllocateData(nComponents);
	OPTFLOAT LastLogL;

	for(int i=0;i<nComponents;i++) {
		nIntems=fread(C.data(),nCols*nCols,sizeof(OPTFLOAT),pF);
		nIntems=fread(M.data(),nCols,sizeof(OPTFLOAT),pF);
		nIntems=fread(&(Weights[i]),sizeof(OPTFLOAT),1,pF);
		Densities[i]->Init(C,M);
	}
	nIntems=fread(&LastLogL,sizeof(OPTFLOAT),1,pF);
}


