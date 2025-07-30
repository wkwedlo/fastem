/*
 * MPIJainEMInitializer.cpp
 *
 *  Created on: Mar 27, 2017
 *      Author: wkwedlo
 */
#include <mpi.h>

#include "../Util/Rand.h"
#include "MPIJainEMInitializer.h"


MPIJainEMInitializer::MPIJainEMInitializer(DistributedMultithreadedDataset &D,int nC,const char *fn) : EMInitializer(D,nC), fname(fn) {
}
void MPIJainEMInitializer::Init(GaussianMixture &G) {
	int nCols=Data.GetColCount();
	long nTotalRows=Data.GetTotalRowCount();

	int Rank,Size;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);

	DynamicArray<float> FlatVector(ncols*Components);
	DynamicArray<long> ObjNums(Components);
	if (Rank==0) {
		for(int i=0;i<Components;i++)
			ObjNums[i]=(long)(Rand()*nTotalRows);
	}
	MPI_Bcast(ObjNums.GetData(),Components,MPI_LONG,0,MPI_COMM_WORLD);
	MultithreadedDataset::LoadSelectedRows(fname,FlatVector,ObjNums);
	int Counter=0;
	for(int i=0;i<Components;i++) {
		for(int j=0;j<ncols;j++) {
			Means.row(i)[j]=FlatVector[Counter++];
		}
	}

	double Weight=1.0/(double)Components;
	double Cov=Data.GetCovMatrixTrace()/(double)ncols/10.0;
	//TRACE1("MPIJainEMInitializer::Init global variance: %g",Cov);
	for(int i=0;i<Components;i++) {
			Weights[i]=Weight;
			Covs[i]=EigenMatrix::Identity(ncols,ncols)*Cov;

	}
	G.Init(Weights,Covs,Means);
}
