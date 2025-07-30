#ifndef KMEANSINITIALIZER_
#define KMEANSINITIALIZER_
#include "../Util/MultithreadedDataset.h"
#include "CentroidVector.h"


class KMeansInitializer  {
	
protected:
	int ncols;
	MultithreadedDataset &Data;
	CentroidVector &CV;
	int nclusters;
	
public:
	KMeansInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl);
	void RepairCentroid(DynamicArray<OPTFLOAT> &vec,int clnum);
	virtual void Init(DynamicArray<OPTFLOAT> &arr)=0;
	virtual ~KMeansInitializer() {};
};

class ForgyInitializer : public KMeansInitializer {
public:
	void Init(DynamicArray<OPTFLOAT> &v);
	ForgyInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl);
};

class MinDistInitializer : public KMeansInitializer {

	int Trials;
public:
	void Init(DynamicArray<OPTFLOAT> &v);
	MinDistInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl,int Tr);
};

class RandomInitializer : public KMeansInitializer {

	DynamicArray<long> PartCounts;
	
public:
	void Init(DynamicArray<OPTFLOAT> &v);
	RandomInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl);
};


class FileInitializer : public KMeansInitializer {
protected:
	DynamicArray<OPTFLOAT> vec;
public:
	void Init(DynamicArray<OPTFLOAT> &v);
	FileInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl,const char *fname);
};
#endif /*KMEANSINITIALIZER_*/
