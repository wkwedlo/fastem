#ifndef KMALGORITHM_H_
#define KMALGORITHM_H_

#include "CentroidVector.h"
#include "CentroidRepair.h"
#include "KMeansInitializer.h"
#include "../Util/MultithreadedDataset.h"
#include "../Util/LargeVector.h"
#include "../Util/Rand.h"

#include "KMeansWithoutMSE/KMeansWithoutMSE.h"
#include "OpenMPKMAReducer.h"
#include "KMeansReportWriter.h"



#ifndef OMPDYNAMIC
#define OMPDYNAMIC
#endif

/// The abstract class representing the K-means algorithm

/**
 * The solution vector (vec parameter of RunKMeans) stores coordinates of the cluster centroids
 * in the following manner
 * (coordinates of the first centroid,coordinates of the second centroid, ....,coordinates of the last centroid).
 */

class KMeansAlgorithm
{
protected:
	int ncols;     		/// Number of columns in dataset
	int nclusters; 		/// Number of clusters K
	int IterCount;      /// K-means iteration counter
	CentroidVector &CV;
	MultithreadedDataset &Data;  /// MultithreadedDataset used to train K-Means
	CentroidRepair *pRepair;
	KMeansReportWriter *pReportWriter;
	int Verbosity;

	/// Random perturbation of the solution vector used by k-means with stochastic relaxation
	void PeturbVector(DynamicArray<OPTFLOAT> &vec, double rtime,double MaxTime);

	/// Init auxiliary data structures used by the algorithm given initial centroid coordinates
	virtual void InitDataStructures(DynamicArray<OPTFLOAT> &vec) {}

	virtual void PrintIterInfo(int i, double BestFit, double Rel,double Avoided, double iterTime);
	virtual void PrintAvgAvoidance(double Avoided);

public:
	KMeansAlgorithm(CentroidVector &aCV,MultithreadedDataset &Data,CentroidRepair *pR);
	/// Runs the KMeans algorithm (with default MinRel until termination)
	virtual double RunKMeans(DynamicArray<OPTFLOAT> &vec,int verbosity,double MinRel=-1,int MaxIter=0);

	/// Runs the KMeans algorithm with stochastic relaxation
	virtual double RunKMeansWithSR(DynamicArray<OPTFLOAT> &vec, bool print,double MaxTime);

	/// single iteration of the K-Means
	virtual double ComputeMSEAndCorrect(DynamicArray<OPTFLOAT> &vec, long *distanceCount=NULL)=0;

	virtual void PrintNumaLocalityInfo() {}
	int GetIterCount() const {return IterCount;}
	int GetNCols() const {return ncols;}
	int GetNClusters() const {return nclusters;}
	void SetReportWriter(KMeansReportWriter *pR) {pReportWriter=pR;}
	void ResetIterCount() {IterCount=0;}
	virtual ~KMeansAlgorithm() {}
};



/// Local data (centers and counts) for each OpenMP thread
struct NaiveThreadData {
	ThreadPrivateVector<OPTFLOAT> row;
};

/// Naive straightforward version of the K-Means algorithm


/**
 * This is the naive (based on definition) version of the K-Means algorithm.
 * However, this version is very well parallelized for NUMA architectures,
 * using OpenMP. It scales very well (tested on the 64-core mordor3 server)
 */
class NaiveKMA : public KMeansAlgorithm, public KMeansWithoutMSE {
protected:

	/// The centroids of the new clusters

	/**
	 * New cluster centroids computed by ComputeMSEAndCorrect method stored as
	 * coordinates of the first centroid, second centroid, ..., k-th centroid
	 */
	ThreadPrivateVector<OPTFLOAT> Center;

	 /// Number of objects allocated to each center (needed to update centroid coordinates).
	ThreadPrivateVector<long> Counts;

	DynamicArray<NaiveThreadData> OMPData;

	LargeVector<int> Assignment;


	OpenMPKMAReducer *pOMPReducer;

	/// Future Data reduction for version MPI, now empty
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<long> &Counts,EXPFLOAT &Fit) {}
	virtual void ReduceMPIData(ThreadPrivateVector<OPTFLOAT> &Centers,ThreadPrivateVector<long> &Counts,bool &bCont) {}
public:
	virtual void PrintNumaLocalityInfo();
	virtual bool CorrectWithoutMSE(DynamicArray<OPTFLOAT> &vec, long *distanceCount);
	virtual void InitDataStructures(DynamicArray<OPTFLOAT> &vec);

	virtual double ComputeMSEAndCorrect(DynamicArray<OPTFLOAT> &vec, long *distanceCount=NULL);
	NaiveKMA(CentroidVector &aCV,MultithreadedDataset &Data,CentroidRepair *pR);
	~NaiveKMA();
};

#endif /*KMALGORITHM_H_*/
