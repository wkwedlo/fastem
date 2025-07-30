/*
 * NumaAlloc.h
 *
 *  Created on: Mar 18, 2015
 *      Author: wkwedlo
 */

#ifndef UTIL_NUMAALLOC_H_
#define UTIL_NUMAALLOC_H_


#include <numaif.h>
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#include <stdint.h>

#include "OpenMP.h"
#include "Array.h"
#include "../EM2/BlockManager.h"
#include "StdOut.h"

class NUMAAllocator {
protected:
    static NUMAAllocator *Instance;

public:
    static NUMAAllocator *GetInstance() {return Instance;}
    static void CreateNUMAAllocator(const char *name=nullptr,const char *param=nullptr);
    virtual void *Alloc(size_t Size)=0;
    virtual void Free( void *ptr,size_t Size)=0;
    virtual void PrintInfo()=0;
    virtual ~NUMAAllocator() {}
};



class FirstTouchNUMAAllocator : public NUMAAllocator {

	double HugePageThr=1.0;
	void HugePagesAllowHeuristic(void *ptr,size_t Size);

public:
	virtual void* Alloc(size_t Size);
	virtual void Free(void *ptr,size_t Size);
    virtual void PrintInfo();
	void SetHugePageThreshold(double Thr) {HugePageThr=Thr;}
};

class ExplicitNUMAAllocator : public NUMAAllocator {
public:
    virtual void* Alloc(size_t Size);
    virtual void Free(void *ptr,size_t Size);
    virtual void PrintInfo();
};


class MallocNUMAAllocator : public NUMAAllocator {
public:
    virtual void* Alloc(size_t Size);
    virtual void Free(void *ptr,size_t Size);
    virtual void PrintInfo();
};


extern inline void* align_pointer_to_page(void* ptr) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned_addr = addr & ~(uintptr_t)(page_size - 1);
    return (void*)aligned_addr;
}

extern inline int NodeOfAddr(void *ptr) {
	int numa_node = -1;
    void *ptr2 = align_pointer_to_page(ptr);
    int rc = numa_move_pages(0, 1, &ptr2, NULL, &numa_node, 0);
	if (rc==-1)
        return -1;
    return numa_node;

}


template <typename Array> double NumaLocalityofPrivateArray (Array & arr) {
	int Size=arr.GetSize();
	int DesiredCount=0;
	int CPU=sched_getcpu();
	int DesiredNode=numa_node_of_cpu(CPU);

	for(int i=0;i<Size;i++) {
		void *ptr=&(arr[i]);
		int RealNode=NodeOfAddr(ptr);
		if (DesiredNode==RealNode)
			DesiredCount++;
	}
	return (double)DesiredCount/(double)Size;
}

template <typename Matrix> double NumaLocalityofPrivateMatrix (Matrix & M) {

	int nRows=M.GetRowCount();
	int DesiredCount=0;
	int CPU=sched_getcpu();
	int DesiredNode=numa_node_of_cpu(CPU);

	for(int i=0;i<nRows;i++) {
		void *ptr=M(i);
		int RealNode=NodeOfAddr(ptr);
		if (DesiredNode==RealNode)
			DesiredCount++;
	}

	return (double)DesiredCount/(double)nRows;
}



template <typename Array> double NumaLocalityofArray (Array & arr) {

	int Size=arr.GetSize();
	int DesiredCount=0;

#pragma omp parallel default(none) firstprivate(Size) reduction(+:DesiredCount) shared(arr)
	{
		int CPU=sched_getcpu();
		int DesiredNode=numa_node_of_cpu(CPU);
#pragma omp for
		for(int i=0;i<Size;i++) {
			void *ptr=&(arr[i]);
			int RealNode=NodeOfAddr(ptr);
			if (DesiredNode==RealNode)
				DesiredCount++;
		}
	}
	return (double)DesiredCount/(double)Size;
}


template <typename LargeArray> double NumaLocalityofLargeArray (LargeArray & A, int nBlocks) {
    long nRows = A.GetRowCount();
    int DesiredCount = 0;
	BlockManager BM(nRows,nBlocks);
#pragma omp parallel default(none) firstprivate(nRows,nBlocks) reduction(+:DesiredCount) shared(A,BM)
    {
        int CPU = sched_getcpu();
        int DesiredNode = numa_node_of_cpu(CPU);

    	for (int k = 0; k < nBlocks; k++) {
        	long BlockStart,BlockRows;
        	BM.GetBlock(k,BlockStart,BlockRows);
        	long Offset,Count;
        	BM.GetThreadSubblock(BlockStart,BlockRows,Offset,Count);
            for (long i = 0; i < Count; i++) {
                void *ptr = A(i + Offset + BlockStart);
                int RealNode = NodeOfAddr(ptr);
                if (DesiredNode == RealNode)
                    DesiredCount++;
            }
        }
    }
    return (double)DesiredCount/(double)nRows;
}

#endif /* UTIL_NUMAALLOC_H_ */
