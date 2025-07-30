/*
 * DistributedMultithreadedDataset.h
 *
 *  Created on: Oct 8, 2015
 *      Author: wkwedlo
 */

#ifndef MPIKMEANS_DISTRIBUTEDNUMADATASET_H_
#define MPIKMEANS_DISTRIBUTEDNUMADATASET_H_

#include "../Util/MultithreadedDataset.h"

class DistributedMultithreadedDataset: public MultithreadedDataset {
protected:

	DynamicArray<long> Offsets;
	DynamicArray<long> Counts;
	void ComputeDataPositions(int Size);
	virtual void ReduceSumStatistics(DynamicArray<OPTFLOAT> &Vec);

public:
	virtual void ReduceStatisticsStep1(DynamicArray<OPTFLOAT> &MeanSum,DynamicArray<float> &Min,DynamicArray<float> &Max);
	virtual void ReduceStatisticsStep2(DynamicArray<OPTFLOAT> &StdSum);

	virtual void GlobalFetchRow(long Position,DynamicArray<float> &row);
	virtual void Load(char *fname);
	DistributedMultithreadedDataset(int anBlocks=1);
	static void QueryHeader(char *fname,long &nRows,int &nCols);
	virtual ~DistributedMultithreadedDataset() {}
};

#endif /* MPIKMEANS_DISTRIBUTEDNUMADATASET_H_ */
