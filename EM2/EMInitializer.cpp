/*
 * EMInitializer.cpp
 *
 *  Created on: Apr 28, 2010
 *      Author: wkwedlo
 */

#include <stdexcept>
#include "../Util/Rand.h"
#include "../Util/StdOut.h"
#include "../Clust/KMeansOrOrInitializer.h"
#include "EMInitializer.h"


EMInitializer::EMInitializer(MultithreadedDataset &D,int nC) : Data(D) {
	Components=nC;
	ncols=D.GetColCount();

	Covs.SetSize(Components);
	Weights.SetSize(Components);
	Means.resize(Components,ncols);
	for(int i=0;i<Components;i++) {
		Covs[i].resize(ncols,ncols);
	}
}

EMInitializer::~EMInitializer() {
}


JainEMInitializer::JainEMInitializer(MultithreadedDataset &D,int nC) : EMInitializer(D,nC) {

}

JainEMInitializer::~JainEMInitializer() {

}

void JainEMInitializer::PrintInfo() {
	kma_printf("Jain EM Initializer\n");
}

void JainEMInitializer::Init(GaussianMixture &G) {


	OPTFLOAT Weight=1.0/(OPTFLOAT)Components;
	OPTFLOAT Cov=Data.GetCovMatrixTrace()/(OPTFLOAT)ncols/10.0;
	DynamicArray<long> ObjNums(Components);

	for(int i=0;i<Components;i++) {

		// Initialize mean component
		long Idx=Rand()*Data.GetRowCount();
		bool found=false;

		do {
			long Idx=Rand()*Data.GetRowCount();
			found=false;
			for(int j=0;j<i;j++)
				if (ObjNums[j]==Idx) {
					found=true;
					break;
				}
			} while(found);
			ObjNums[i]=Idx;

		const float *row=Data.GetRowNew(Idx);

		Weights[i]=Weight;

		for(int j=0;j<ncols;j++)
			Means.row(i)[j]=row[j];

		Covs[i]=EigenMatrix::Identity(ncols,ncols)*Cov;

	}

	G.Init(Weights,Covs,Means);

}

KMeansEMInitializer::KMeansEMInitializer(MultithreadedDataset &D,int nC) : EMInitializer(D,nC) {
	pCV=new CentroidVector(nC,D.GetColCount());
	pRepair=new CentroidRandomRepair(D,nC);
	pKMAlg=new NaiveKMA(*pCV,D,pRepair);
	pKMInit=new ForgyInitializer(D,*pCV,nC);
	KMVec.SetSize(Components*ncols);
	ClassNums.SetSize(D.GetRowCount());
	ObjCounts.SetSize(nC);
}

KMeansEMInitializer::~KMeansEMInitializer() {
	delete pCV;
	delete pRepair;
	delete pKMAlg;
	delete pKMInit;
}

void KMeansEMInitializer::PrintInfo() {
	kma_printf("K-means EM initializer\n");
}

void KMeansEMInitializer::Init(GaussianMixture &G) {
	pKMInit->Init(KMVec);
	pKMAlg->RunKMeans(KMVec,0,1e-4,30);
	pCV->ClassifyDataset(KMVec,Data,ClassNums);
	long nRows=Data.GetRowCount();

	for(int i=0;i<Components;i++) {
		for(int j=0;j<ncols;j++)
			Means.row(i)[j]=KMVec[i*ncols+j];
		ObjCounts[i]=0;
		Covs[i]=EigenMatrix::Zero(ncols,ncols);
	}

	EigenRowVector row(ncols);

	for(long i=0;i<nRows;i++) {
		int Class=ClassNums[i];
		ObjCounts[Class]++;
		for(int j=0;j<ncols;j++)
			row[j]=Data.GetRowNew(i)[j]-Means.row(Class)[j];

		Covs[Class].noalias()+=row.transpose()*row;
	}

	for(int i=0;i<Components;i++) {
		OPTFLOAT f=1.0/((OPTFLOAT)ObjCounts[i]-1.0);
		Covs[i]*=f;
		Weights[i]=(OPTFLOAT)ObjCounts[i]/(OPTFLOAT)nRows;
	}
	G.Init(Weights,Covs,Means);
	//G.Dump();
}

FileEMInitializer::FileEMInitializer(MultithreadedDataset &D,int nC,const char *fname) :EMInitializer(D,nC) {
	pF=fopen(fname,"rb");
	if (pF==nullptr)
		throw std::invalid_argument("Cannot open file with Gaussian mixture");

}

FileEMInitializer::~FileEMInitializer() {
	if (pF!=nullptr)
		fclose(pF);
}

void FileEMInitializer::Init(GaussianMixture &G) {
	GaussianMixture G1(pF);
	G=G1;
}

void FileEMInitializer::PrintInfo() {
	kma_printf("File EM initializer\n");

}

KMeansOrOrEMInitializer::KMeansOrOrEMInitializer(MultithreadedDataset &D,int nC) : KMeansEMInitializer(D,nC) {
	delete pKMInit;
	pKMInit = new KMeansOrOrInitializer(D,*pCV,nC,1.4);
}

void KMeansOrOrEMInitializer::PrintInfo() {
	kma_printf("K-means|| EM initializer\n");

}

EMInitializer *CreateEMInitializer(const char *name,MultithreadedDataset &Data,int nC) {
	if (name==NULL || !strcmp(name,"jain"))
		return new JainEMInitializer(Data,nC);
	if (!strcmp(name,"kmeans"))
		return new KMeansEMInitializer(Data,nC);
	if (!strcmp(name,"kmeansoror"))
		return new KMeansOrOrEMInitializer(Data,nC);
	return new FileEMInitializer(Data,nC,name);

}
