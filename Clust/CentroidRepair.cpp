/*
 * CentroidRepair.cpp
 *
 *  Created on: Feb 3, 2016
 *      Author: wkwedlo
 */

#include <stdio.h>
#include "../Util/Rand.h"
#include "../Util/StdOut.h"
#include "CentroidRepair.h"

CentroidRepair::CentroidRepair(int ncl){
	nclusters=ncl;
}

CentroidRandomRepair::CentroidRandomRepair(MultithreadedDataset &D,int ncl) : Data(D),CentroidRepair(ncl) {

}

void CentroidRandomRepair::RepairVec(DynamicArray<OPTFLOAT> &vec,int Pos) {
	long Source=Rand()*Data.GetTotalRowCount();
	int nCols=Data.GetColCount();
	const float *row=Data.GetRowNew(Source);
	for(int i=0;i<nCols;i++)
		vec[Pos*nCols+i]=row[i];
	kma_printf("Empty centroid %d repaired by row %ld\n",Pos,Source);
}

CentroidDeterministicRepair::CentroidDeterministicRepair(MultithreadedDataset &D,int ncl) : Data(D),CentroidRepair(ncl) {
	Next=0;
}

void CentroidDeterministicRepair::RepairVec(DynamicArray<OPTFLOAT> &vec,int Pos) {
	int nCols=Data.GetColCount();
	const float *row=Data.GetRowNew(Next);
	for(int i=0;i<nCols;i++)
		vec[Pos*nCols+i]=row[i];
	kma_printf("Empty centroid %d repaired by row %ld\n",Pos,Next);
	Next++;
	if (Next==Data.GetTotalRowCount())
		Next=0;

}

