#ifndef __LARGEMATRIX_H
#define __LARGEMATRIX_H



#include "Matrix.h"

#ifdef _OPENMP
#include <omp.h>
#endif

extern bool numa;

/// class for very large matrices accessed from multiple threads on a NUMA system
template <typename T> class LargeMatrix : public MatrixBase<T> {

protected:
	using MatrixBase<T>::nRows;
	using MatrixBase<T>::nCols;
	using MatrixBase<T>::nStride;
	using MatrixBase<T>::pBuff;

private:
	LargeMatrix(const LargeMatrix<T> *M);

public:
	using MatrixBase<T>::GetData;
	void SetSize(long nR,long nC,long Alignment=1,bool bInit=true);
	LargeMatrix();
    LargeMatrix(long nR,long nC,long Alignment=1,bool bInit=true);
    ~LargeMatrix();
};


template <typename T> void LargeMatrix<T>::SetSize(long nR,long nC,long Alignment,bool bInit) {

	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	if (pBuff!=NULL) {
		pAlloc->Free(pBuff,(long)nRows*(long)nStride*sizeof(T));
	}
	nRows=nR;
	nCols=nC;

	nStride=nCols;
	long div=nStride%Alignment;
	if (div)
		nStride+=(Alignment-div);

	pBuff=(T *)pAlloc->Alloc((long)nRows*(long)nStride*sizeof(T));
    if (bInit) {
#pragma omp parallel for default(none)
        for (long i = 0; i < nRows; i++) {
            for (long j = 0; j < nCols; j++)
                (*this)(i, j) = (T) 0;
        }
    }
}

template <typename T> LargeMatrix<T>::LargeMatrix(long nR,long nC,long Alignment,bool bInit) {
	pBuff=NULL;
	SetSize(nR,nC,Alignment,bInit);
}

template <typename T> LargeMatrix<T>::LargeMatrix() {
	pBuff=NULL;
	nStride=nRows=nCols=0;
}


template <typename T> LargeMatrix<T>::~LargeMatrix() {
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pAlloc->Free(pBuff,(long)nRows*(long)nStride*sizeof(T));
}

#endif
