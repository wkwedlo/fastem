#ifndef KMEANSWITHOUTMSE_H_
#define KMEANSWITHOUTMSE_H_

#include "../../Util/Array.h"
#include "../CentroidVector.h"

class KMeansWithoutMSE {
public:
	KMeansWithoutMSE(CentroidVector &CV, MultithreadedDataset &Data, int &IterCount);

	virtual ~KMeansWithoutMSE() { };

	/**
	 * Run the k-means algorithm without calculating the MSE.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 *  verbosity - debug info verbosity
	 */
	void RunKMeansWithoutMSE(DynamicArray<OPTFLOAT> &vec, const int verbosity,int MaxIter);

	/**
	 * A single k-means iteration without MSE calculation.
	 *  vec - centroid vector [1-st centroid, 2-nd centroid, ...]
	 *  distanceCount - number of distance calculations
	 */
	virtual bool CorrectWithoutMSE(DynamicArray<OPTFLOAT> &vec, long *distanceCount) = 0;

	/**
	 * Init auxiliary data structures used by the algorithm
	 * given initial centroid coordinates.
	 *  vec - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 */
	virtual void InitDataStructures(DynamicArray<OPTFLOAT> &vec) = 0;

private:
	CentroidVector &cv;
	MultithreadedDataset &data;  /// MultithreadedDataset used to train K-Means
	int &iterCount;
};

#endif /* KMEANSWITHOUTMSE_H_ */
