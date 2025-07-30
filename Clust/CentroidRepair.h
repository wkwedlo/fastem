/*
 * CentroidRepair.h
 *
 *  Created on: Feb 3, 2016
 *      Author: wkwedlo
 */

#ifndef CLUST_CENTROIDREPAIR_H_
#define CLUST_CENTROIDREPAIR_H_

#include "../Util/MultithreadedDataset.h"

class CentroidRepair {
protected:
	int nclusters;
public:
	virtual void RepairVec(DynamicArray<OPTFLOAT> &vec,int Pos)=0;
	explicit CentroidRepair(int ncl);
	virtual ~CentroidRepair()=default;
};


class CentroidRandomRepair : public CentroidRepair {
protected:
	MultithreadedDataset &Data;
public:
	CentroidRandomRepair(MultithreadedDataset &Data,int ncl);
	virtual void RepairVec(DynamicArray<OPTFLOAT> &vec,int Pos);
};

class CentroidDeterministicRepair : public CentroidRepair {
protected:
	MultithreadedDataset &Data;
	long Next;
public:
	CentroidDeterministicRepair(MultithreadedDataset &Data,int ncl);
	virtual void RepairVec(DynamicArray<OPTFLOAT> &vec,int Pos);
};

#endif /* CLUST_CENTROIDREPAIR_H_ */
