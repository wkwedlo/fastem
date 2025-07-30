#ifndef CENTROIDVECTOR_H_
#define CENTROIDVECTOR_H_

#include "../Util/Array.h"
#include "../Util/MultithreadedDataset.h"
#include "../Util/LargeVector.h"

/// Class representing operations on vector of centroids


/**This class encapsulates operations on a vector (vec or v method parameter)
 * storing coordinates of cluster centroids in the following order:
 * \member ncols coordinates of the first centroid, ncols coordinates of the
 * second centroids, ..., ncols coordinates of the centroid no nclusters.
 *
 * So the size of vec or v is always ncols*clusters;
 */


class CentroidVector
{
	/// number of centroids (clusters)
	int nclusters;
	/// number of columns in the data set, i.e. dimension of the feature space
	int ncols;
public:
	/** \param ncl is the number of clusters (centroids)
	 *  \param ncol is number of coordinates of each centroid i.e. dimension of the feature space
	 */
	CentroidVector(int ncl,int ncol) {nclusters=ncl;ncols=ncol;}
	void ClassifyDataset(DynamicArray<OPTFLOAT> &v,const MultithreadedDataset &Data, DynamicArray<int> &ClNums);
	void ClassifyDataset(DynamicArray<OPTFLOAT> &v,const MultithreadedDataset &Data, DynamicArray<int> &ClNums,DynamicArray<OPTFLOAT> &Distances);
	void FindObjectsInCluster(const DynamicArray<OPTFLOAT> &vec,const MultithreadedDataset &Data,int Num,
			DynamicArray<long> &ObjNums);

	void Sort(DynamicArray<OPTFLOAT> &vec);
	int GetNCols() {return ncols;}
	int GetNClusters() {return nclusters;}
	double ComputeMSE(DynamicArray<OPTFLOAT> &vec, const MultithreadedDataset &Data);
	double ComputeMSE(DynamicArray<OPTFLOAT> &vec, const MultithreadedDataset &Data, const DynamicArray<int> &Assignment);
	void ComputeCenters(DynamicArray<OPTFLOAT> &vec, const MultithreadedDataset &Data, const DynamicArray<int> &Nums );
	void ComputeClusterSSE(const DynamicArray<OPTFLOAT> &vec, const MultithreadedDataset &Data,DynamicArray<OPTFLOAT> &ClustSSE);
	

	
	inline void ConvertToOptFloat(ThreadPrivateVector<OPTFLOAT> &vec,const float * __restrict__ row) {
		for(int i=0;i<ncols;i++)
			vec[i]=(OPTFLOAT)row[i];
	}

	inline void AddRow(const int k,ThreadPrivateVector<OPTFLOAT> &v,const ThreadPrivateVector<OPTFLOAT> &row) {
		const int start=k*ncols;
		OPTFLOAT * __restrict__ vptr=v.GetData()+start;
		const OPTFLOAT * __restrict__ rptr=row.GetData();

#pragma omp simd
		for(int i=0;i<ncols;i++)
			vptr[i]+=rptr[i];

	}

	inline void MoveRow(const int Src,const int Dst,ThreadPrivateVector<OPTFLOAT> &v,const ThreadPrivateVector<OPTFLOAT> &row) {
		const int StartDst=Dst*ncols;
		const int StartSrc=Src*ncols;

		OPTFLOAT * __restrict__ ptrSrc=v.GetData()+StartSrc;
		OPTFLOAT * __restrict__ ptrDst=v.GetData()+StartDst;
		const OPTFLOAT * __restrict__ rptr=row.GetData();

#pragma omp simd
		for(int i=0;i<ncols;i++) {
			ptrSrc[i]-=rptr[i];
			ptrDst[i]+=rptr[i];
		}
	}



	inline OPTFLOAT SquaredDistance(const float * __restrict__ row1,const float * __restrict__ row2) {
		OPTFLOAT ssq=(OPTFLOAT)0.0;

#pragma omp simd reduction(+:ssq)
		for(int i=0;i<ncols;i++)
			ssq+=((row2[i]-row1[i])*(row2[i]-row1[i]));
		return ssq;

	}


	inline OPTFLOAT SquaredDistance(const int k,const DynamicArray<OPTFLOAT> &v,
					const ThreadPrivateVector<OPTFLOAT> &row)
			{
				const int start=k*ncols;
				const OPTFLOAT * __restrict__ vptr=v.GetData()+start;
				const OPTFLOAT * __restrict__ rptr=row.GetData();
				OPTFLOAT ssq=(OPTFLOAT)0.0;

				#pragma omp simd reduction(+:ssq)
				for(int i=0;i<ncols;i++)
					ssq+=((vptr[i]-rptr[i])*(vptr[i]-rptr[i]));
				return ssq;
			}

	/// Faster version, computes squared Euclidean distance between a training vector row and k-th centroid
	inline OPTFLOAT SquaredDistance(const int k,const DynamicArray<OPTFLOAT> &v,
						const float * __restrict__ row)
				{
					const int start=k*ncols;
					const OPTFLOAT * __restrict__ vptr=v.GetData()+start;
					OPTFLOAT ssq=(OPTFLOAT)0.0;

					#pragma omp simd reduction(+:ssq)
					for(int i=0;i<ncols;i++)
						ssq+=((vptr[i]-row[i])*(vptr[i]-row[i]));
					return ssq;
				}
#ifdef DOUBLEFLOAT
	inline OPTFLOAT SquaredDistance(const int k,const DynamicArray<OPTFLOAT> &v,
						const OPTFLOAT * __restrict__ row)
				{
					const int start=k*ncols;
					const OPTFLOAT * __restrict__ vptr=v.GetData()+start;
					OPTFLOAT ssq=(OPTFLOAT)0.0;

					#pragma omp simd reduction(+:ssq)
					for(int i=0;i<ncols;i++)
						ssq+=((vptr[i]-row[i])*(vptr[i]-row[i]));
					return ssq;
				}

	inline OPTFLOAT SquaredDistance(const OPTFLOAT * __restrict__ vptr,const float * __restrict__ row)
					{
						OPTFLOAT ssq=(OPTFLOAT)0.0;
						#pragma omp simd reduction(+:ssq)
						for(int i=0;i<ncols;i++)
							ssq+=((vptr[i]-row[i])*(vptr[i]-row[i]));
						return ssq;
					}

#endif

	/**
	 * Compute the distance from centroid to centroid.
	 *  v1 - the first centroid vector [1-st centroid, 2-nd centroid, ...]
	 *  v2 - the second centroid vector [1-st centroid, 2-nd centroid, ...]
	 *  c - the centroid index
	 */
	inline OPTFLOAT SquaredDistance(const DynamicArray<OPTFLOAT> &v1, const DynamicArray<OPTFLOAT> &v2, const int c) {
		const int s = c * ncols;

		const OPTFLOAT * __restrict__ v1ptr=v1.GetData()+s;
		const OPTFLOAT * __restrict__ v2ptr=v2.GetData()+s;
		OPTFLOAT ssq = (OPTFLOAT) 0.0;

		#pragma omp simd reduction(+:ssq)
		for(int i=0;i<ncols;i++)
			ssq += ((v1ptr[i] - v2ptr[i]) * (v1ptr[i] - v2ptr[i]));
		return ssq;
	}

	inline OPTFLOAT PartialSquaredDistance(const int k,const DynamicArray<OPTFLOAT> &v,
				const float *row,const OPTFLOAT best)
		{
			int start=k*ncols;
			OPTFLOAT ssq=(OPTFLOAT)0.0;
			
			#pragma ivdep
		//	#pragma vector aligned
			for(int i=0;i<ncols && ssq<best;i++,start++) 
				ssq+=((v[start]-row[i])*(v[start]-row[i]));
			return ssq;
		}
	
	inline OPTFLOAT CentroidSquaredDistance(const DynamicArray<OPTFLOAT> &v,const int c1,const int c2) {
		const int s1=c1*ncols;
		const int s2=c2*ncols;

		const OPTFLOAT * __restrict__ s1ptr=v.GetData()+s1;
		const OPTFLOAT * __restrict__ s2ptr=v.GetData()+s2;

		OPTFLOAT ssq=(OPTFLOAT)0.0;

		#pragma omp simd reduction(+:ssq)
		for(int i=0;i<ncols;i++)
			ssq+=((s1ptr[i]-s2ptr[i])*(s1ptr[i]-s2ptr[i]));
		return ssq;
		
	}

	/**
	 * Compute the norm of a given centroid.
	 * v - the centroid vector [1-st centroid, 2-nd centroid, ...]
	 * c - the centroid index in the vector
	 */
	inline OPTFLOAT SquaredCentroidNorm(const DynamicArray<OPTFLOAT> &v, const int c) {
		const int s = c * ncols;

		const OPTFLOAT * __restrict__ ptr = v.GetData() + s;
		OPTFLOAT ssq = (OPTFLOAT) 0.0;

		#pragma omp simd reduction(+:ssq)
		for (int i = 0; i < ncols; i++)
			ssq += ptr[i] * ptr[i];
		return ssq;
	}

	inline OPTFLOAT ComputeDot(const OPTFLOAT * __restrict__ row1,const OPTFLOAT * __restrict__ row2) {
		OPTFLOAT sum = (OPTFLOAT) 0.0;

		#pragma omp simd reduction(+:sum)
		for (int i = 0; i < ncols; i++)
			sum += row1[i] * row2[i];
		return sum;
	}


	/**
	 * Compute the norm of a given row.
	 * row - the row
	 */
	inline OPTFLOAT SquaredRowNorm(const float * __restrict__ row) {
		OPTFLOAT ssq = (OPTFLOAT) 0.0;

		#pragma omp simd reduction(+:ssq)
		for (int i = 0; i < ncols; i++)
			ssq += (row[i] * row[i]);
		return ssq;
	}
};

#endif /*CENTROIDVECTOR_H_*/
