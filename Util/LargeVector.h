/*
 * LargeVector.h
 *
 *  Created on: Mar 18, 2015
 *      Author: wkwedlo
 */

#ifndef UTIL_LARGEVECTOR_H_
#define UTIL_LARGEVECTOR_H_

#include <sys/mman.h>
#include "Debug.h"
#include "NumaAlloc.h"


template <typename T> class VectorBase {
protected:
	T * __restrict__ pBuff;
	long Size;
	void DestroyBuffer();
	virtual void AllocBuffer(long aSize)=0;

public:
	T *GetData() {return pBuff;}
	const T *GetData() const {return pBuff;}
    long GetSize() const {return Size;}
	const T & operator[](long i) const;
    T & operator[](long i);
	void SetSize(long aSize);
	virtual ~VectorBase();

};

template <typename T> void VectorBase<T>::DestroyBuffer() {
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pAlloc->Free(pBuff,(long)Size*sizeof(T));
	pBuff=NULL;
	Size=0;
}

template <typename T> VectorBase<T>::~VectorBase() {
	if (pBuff!=NULL)
		DestroyBuffer();
}

template <typename T> void VectorBase<T>::SetSize(long aSize) {
	if (pBuff!=NULL)
		DestroyBuffer();
	AllocBuffer(aSize);
}


template <typename T> inline T &VectorBase<T>::operator[](long i) {
	ASSERT(i>=0);
	ASSERT(i<Size);
	ASSERT(pBuff!=NULL);
	return pBuff[i];
}

template <typename T> inline const T &VectorBase<T>::operator[](long i) const {
	ASSERT(i>=0);
	ASSERT(i<Size);
	ASSERT(pBuff!=NULL);
	return pBuff[i];
}


template <typename T> class LargeVector : public VectorBase<T> {
	using VectorBase<T>::pBuff;
	using VectorBase<T>::Size;
	

protected:
	using VectorBase<T>::DestroyBuffer;
	void AllocBuffer(long aSize);

public:
	using VectorBase<T>::GetData;
	using VectorBase<T>::operator[];
	using VectorBase<T>::GetSize;
	LargeVector() {pBuff=NULL;Size=0;}
	LargeVector &operator=(const LargeVector &vec);
	virtual ~LargeVector() {}
};


template <typename T> void LargeVector<T>::AllocBuffer(long aSize) {
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pBuff=(T *)pAlloc->Alloc((long)aSize*sizeof(T));
	Size=aSize;
#pragma omp parallel for
	for(long i=0;i<Size;i++) {
		pBuff[i]=(T)0;
	}
}

template <typename T> LargeVector<T> &LargeVector<T>::operator=(const LargeVector &vec) {
	ASSERT(Size==vec.Size);
#pragma omp parallel for
	for(long i=0;i<Size;i++) {
		pBuff[i]=vec.pBuff[i];
	}
	return *this;
}

template <typename T> class ThreadPrivateVector : public VectorBase<T> {
	using VectorBase<T>::pBuff;
	using VectorBase<T>::Size;
	using VectorBase<T>::DestroyBuffer;


protected:
	virtual void AllocBuffer(long aSize);
public:
	using VectorBase<T>::GetData;
	using VectorBase<T>::GetSize;
	using VectorBase<T>::operator[];
	ThreadPrivateVector() {pBuff=NULL;Size=0;}
	virtual ~ThreadPrivateVector() {}
};



template <typename T> void ThreadPrivateVector<T>::AllocBuffer(long aSize) {
	NUMAAllocator *pAlloc=NUMAAllocator::GetInstance();
	pBuff=(T *)pAlloc->Alloc((long)aSize*sizeof(T));
	Size=aSize;
	for(long i=0;i<Size;i++)
		pBuff[i]=(T)0;
}

#endif /* UTIL_LARGEVECTOR_H_ */
