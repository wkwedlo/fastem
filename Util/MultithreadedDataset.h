/*
 * MultithreadedDataset.h
 *
 *  Created on: Mar 28, 2013
 *      Author: wkwedlo
 */

#ifndef NUMADATASET_H_
#define NUMADATASET_H_
#include <sys/uio.h>
#include <string>

#include "Optfloat.h"
#include "../Util/Array.h"
#include "../Util/LargeMatrix.h"

/**
 * @brief Class for storing large datasets in memory
 *
 * The class is optimized for parallel access to the data. It is used in parallelized algorithms.
 * The data is stored in a large matrix. The class provides methods for loading and saving the data,
 * standarizing columns, and converting values between standarized and raw values. Data is stored using row-major order.
 */
class MultithreadedDataset {

	LargeMatrix<float> Data;


	struct alignas(64) ThreadInfo {
		DynamicArray<float> Min,Max;
		DynamicArray<OPTFLOAT> MeanSum,StdDevSum;
	};
	DynamicArray<ThreadInfo> ThreadData;
protected:
	long nTotalRows;
    int HeaderSize;
	int nBlocks;
	

	OPTFLOAT TraceCov;

	DynamicArray<float> Max;
	DynamicArray<float> Min;


	// Std and Mean measured from the dataset
	DynamicArray<OPTFLOAT> OrigMeans;
	DynamicArray<OPTFLOAT> OrigStdDevs;
	
	// Std and Mean used in conversion (originally 1 and 0)
	DynamicArray<OPTFLOAT> Means;
	DynamicArray<OPTFLOAT> StdDevs;

    void FindMinMax();
	void PartialLoad(char *fname,long StartRow,long nR,int nC,long _Alignment=4);
	void LoadHeader(int fd,long &nRows,int &nCols);
	void ReadChunks(int fd,float *pData,long ChunkCount,long FirstRow,DynamicArray<struct iovec> &IOVecs);
	virtual void ReduceStatisticsStep1(DynamicArray<OPTFLOAT> &MeanSum,DynamicArray<float> &Min,DynamicArray<float> &Max) {}
	virtual void ReduceStatisticsStep2(DynamicArray<OPTFLOAT> &StdSum) {}
	virtual void ComputeStatistics();


public:
	static void QueryHeader(char *fname,long &nRows,int &nCols);
	static void LoadSelectedRows(const char *fname, DynamicArray<float> &FlatVector, const DynamicArray<long> &ObjNums);


	void LoadAndProcess(char *fname, int rank = 0, int size = 1);
	
	double GetCovMatrixTrace() const {return TraceCov;}

	/// Standarizes column in a dataset 
	void Standarize(int Column);
	
	/// Returns the mean of the original (not standarized) column;	
	float GetMean(int Column) const {return OrigMeans[Column];}
	
	float GetMin(int Column) const {return Min[Column];}

	float GetMax(int Column) const {return Max[Column];}

	/// Returns the standard deviation of the (not standarized) column;
	float GetStdDev(int Column) const {return OrigStdDevs[Column];}
	


	void SwapRows(long i,long j);


	// Index operator e.g. D(i,j) returns element from row i and column j
	virtual float &operator()(long row,int col) {return Data(row,col);}
	// Index operator for read-only access
	virtual const float &operator()(long row,int col) const {return  Data(row,col);}
	/// Number of rows i.e. the learning vectors
	virtual long GetRowCount() const {return Data.GetRowCount();}

	virtual long GetTotalRowCount() const {return nTotalRows;}


	long GetStride() const {return Data.GetStride();}

	
	/// Number of columns i.e. the variables
	virtual int GetColCount() const {return Data.GetColCount();}
	
	const float * GetData() const {return Data.GetData();}
	float * GetData()  {return Data.GetData();}
	
	/// Returns i-th row
	const float *GetRowNew(long i) const {
		return Data(i);
	}
	float *GetRowNew(long i){
		return Data(i);
	}	
	
	/// Loads parts of data from binary file (used in parallelized algorithm)
	//virtual void Load(char *fname,int proc,int nprocs);
	
	virtual void Save(char* fname);
	
	///Just reserves memory:
	long AllocateMemory(long rows, int cols);
	


	double NumaLocality() {return NumaLocalityofLargeArray(Data,nBlocks);}
	virtual void GlobalFetchRow(long Position,DynamicArray<float> &row);
	virtual void Load(char *fname,long _Alignment=4);
	explicit MultithreadedDataset(int anBlocks=1);
	void LoadSelectedObjects(const MultithreadedDataset &Src,const DynamicArray<long> &ObjNums);
	void LoadFromCenterArray(DynamicArray< DynamicArray<float> > &Centers);
	void LoadFromCentroidVector(DynamicArray<OPTFLOAT> &vec,int ncols,int nclusters);

	virtual ~MultithreadedDataset();
};

void ReadAssignment(std::string AssignmentName,NoInitArray<int> &Assignment);

#endif /* NUMADATASET_H_ */
