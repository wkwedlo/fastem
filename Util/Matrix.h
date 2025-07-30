#ifndef UTIL_MATRIX_H_
#define UTIL_MATRIX_H_

#include "Debug.h"
#include "NumaAlloc.h"
#include <stdio.h>


template <typename T> class MatrixBase {

protected:
	long nRows,nCols,nStride;
	T * __restrict__ pBuff;
public:
	T *operator()(long i);
	const T *operator()(long i) const;
	T &operator()(long i,long j);
    const T & operator()(long i,long j) const;

    long GetRowCount() const {return nRows;}
    long GetColCount() const {return nCols;}
    long GetStride() const {return nStride;}

	T *GetData();
	const T *GetData() const;
	MatrixBase();
	virtual ~MatrixBase() {;}
	void fprintf(FILE *pF);
};

template <typename T> MatrixBase<T>::MatrixBase() {
	pBuff=NULL;
	nRows=0;
	nCols=0;
	nStride=0;
}
template <typename T> T *MatrixBase<T>::GetData() {
	ASSERT(pBuff!=NULL);
	return pBuff;
}

template <typename T> const T *MatrixBase<T>::GetData() const {
	ASSERT(pBuff!=NULL);
	return pBuff;
}

template <typename T> T *MatrixBase<T>::operator()(long i) {
	ASSERT(pBuff!=NULL);
	ASSERT(i>=0);
	ASSERT(i<nRows);
	return pBuff+(long)i*nStride;
}

template <typename T> const T *MatrixBase<T>::operator()(long i) const {
	ASSERT(pBuff!=NULL);
	ASSERT(i>=0);
	ASSERT(i<nRows);
	return pBuff+(long)i*nStride;
}


template <typename T> T& MatrixBase<T>::operator()(long i,long j)
{
	ASSERT(pBuff!=NULL);
	ASSERT(i>=0);
	ASSERT(i<nRows);
	ASSERT(j>=0);
	ASSERT(j<nCols);
	return pBuff[(long)i*nStride+(long)j];
}

template <typename T> const T& MatrixBase<T>::operator()(long i,long j) const
{
	ASSERT(i>=0);
	ASSERT(pBuff!=NULL);
	ASSERT(i<nRows);
	ASSERT(j>=0);
	ASSERT(j<nCols);
	return pBuff[(long)i*nStride+(long)j];
}


template <typename T> void MatrixBase<T>::fprintf(FILE *pF) {
	for(long i=0;i<nRows;i++) {
		for(long j=0;j<nCols;j++)
			::fprintf(pF,"%g ",(double)((*this)(i,j)));
		::fprintf(pF,"\n");
	}
}

template <typename T> class Matrix : public MatrixBase<T> {

protected:
	using MatrixBase<T>::nRows;
	using MatrixBase<T>::nCols;
	using MatrixBase<T>::nStride;
	using MatrixBase<T>::pBuff;
public:
    Matrix<T> &operator=(const Matrix<T> &M);

	Matrix(long nR,long nC,long Alignment=1);
    Matrix(long nR,long nC,T *pData,long Alignment=1);
    Matrix(const Matrix<T> &M);
    Matrix();
    void SetSize(long nR,long nC,long Alignment=1);
    void SetZero();
    ~Matrix();
};

template <typename T> void Matrix<T>::SetZero() {
	for(long i=0;i<nRows;i++)
		for(long j=0;j<nCols;j++)
			(*this)(i,j)=(T)0;
}

template <typename T> void Matrix<T>::SetSize(long nR,long nC,long Alignment) {

	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();

	if (pBuff!=NULL)
		pAlloc->Free(pBuff,sizeof(T)*(long)nRows*(long)nStride);


	nRows=nR;
	nCols=nC;

	nStride=nCols*sizeof(T);
	long div=nStride%Alignment;
	if (div)
		nStride+=(Alignment-div);
	nStride/=sizeof(T);


	pBuff=(T *)pAlloc->Alloc((long)nRows*(long)nStride*sizeof(T));
	SetZero();
}


template <typename T> Matrix<T>::Matrix(const Matrix<T> &M) {
	nRows=M.nRows;
	nCols=M.nCols;
	nStride=M.nStride;
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pBuff=(T *)pAlloc->Alloc((long)nRows*(long)nStride*sizeof(T));


	for(long i=0;i<nRows;i++)
		for(long j=0;j<nCols;j++)
			(*this)(i,j)=M(i,j);
}

template <typename T> Matrix<T>::Matrix() {
	nRows=0;
	nCols=0;
	pBuff=0;
	nStride=0;
}

template <typename T> Matrix<T>::Matrix(long nR,long nC,long Alignment) {

	pBuff=NULL;
	SetSize(nR,nC,Alignment);
}

template <typename T> Matrix<T>::Matrix(long nR,long nC,T *pData,long Alignment)  {
	pBuff=NULL;
	SetSize(nR,nC,Alignment);
	for(long i=0;i<nRows;i++)
		for(long j=0;j<nCols;j++)
			(*this)(i,j)=pData[i*nCols+j];
}


template <typename T> Matrix<T>::~Matrix() {
	if (pBuff!=NULL) {
		NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
		pAlloc->Free(pBuff,sizeof(T)*(long)nRows*(long)nStride);
	}
}

template <typename T> Matrix<T>& Matrix<T>::operator=(const Matrix<T> &M) {
	if (&M!=this) {
		NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
		if (pBuff!=NULL) {
			if (nRows!=M.nRows || nStride!=M.nStride  ) {
				pAlloc->Free(pBuff,sizeof(T)*(long)nRows*(long)nStride);
				pBuff=(T *)pAlloc->Alloc((long)M.nRows*(long)M.nStride*sizeof(T));
			}
		} else 	pBuff=(T *)pAlloc->Alloc((long)M.nRows*(long)M.nStride*sizeof(T));



		nRows=M.nRows;
		nCols=M.nCols;
		nStride=M.nStride;

		for(long i=0;i<nRows;i++)
			for(long j=0;j<nCols;j++)
				(*this)(i,j)=M(i,j);

	}
	return *this;
}


#endif /* UTIL_MATRIX_H_ */
