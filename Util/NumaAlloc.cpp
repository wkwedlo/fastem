#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <stdlib.h>
#include <numa.h>
#include <sys/mman.h>
#include "NumaAlloc.h"
#include "OpenMP.h"
#include "Debug.h"
#include "StdOut.h"

NUMAAllocator *NUMAAllocator::Instance;


void NUMAAllocator::CreateNUMAAllocator(const char *name,const char *param) {
    if (Instance)
        delete Instance;
    if (name== nullptr || !std::strcmp(name,"firsttouch")) {
        auto ptr=new FirstTouchNUMAAllocator;
        if (param!= nullptr) {
            double Thr=atof(param);
            ptr->SetHugePageThreshold(Thr);
        }
        Instance=ptr;
    } else if (!std::strcmp(name,"explicit")) {
        Instance = new ExplicitNUMAAllocator;
    } else if (!std::strcmp(name,"malloc")) {
        Instance = new MallocNUMAAllocator;
    } else
        throw std::invalid_argument("Invalid NUMA allocator name");
}


#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE 15
#endif

const int HugePageSize=2*1024*1024;

//#define TRACE_NUMA_ALLOC

void FirstTouchNUMAAllocator::HugePagesAllowHeuristic(void *ptr,size_t Size) {
	int maxnodes=std::min(omp_get_max_threads(),numa_num_configured_nodes());
	double HugePagesPerNode=(double)Size/(double)maxnodes/(double)HugePageSize;
	bool bAllow=false;
	if (maxnodes==1 || HugePagesPerNode>HugePageThr)
		bAllow=true;
#ifdef TRACE_NUMA_ALLOC
	kma_printf("Memory allocation of %f MB, predicted huge pages: %f, forcing no hugepages: %d ",(double)Size/1024.0/1024.0,HugePagesPerNode,!bAllow);
#endif
	if (!bAllow) {
		int Ret=madvise(ptr,Size, MADV_NOHUGEPAGE);
#ifdef TRACE_NUMA_ALLOC
		kma_printf("madvise result: %d\n",Ret);
#endif
	} else  {
#ifdef TRACE_NUMA_ALLOC
		kma_printf("\n");
#endif
	}
}


void* FirstTouchNUMAAllocator::Alloc(size_t Size) {
	void *ptr=mmap(NULL,Size,PROT_READ | PROT_WRITE,MAP_PRIVATE | MAP_ANONYMOUS,-1,0);
	HugePagesAllowHeuristic(ptr,Size);
	return ptr;
}

void FirstTouchNUMAAllocator::Free(void *ptr,size_t Size) {
	munmap(ptr,Size);
}

void FirstTouchNUMAAllocator::PrintInfo() {
    kma_printf("First touch NUMA allocator with hugepage threshold %f\n",HugePageThr);
}


void* ExplicitNUMAAllocator::Alloc(size_t Size) {
    void *ptr=mmap(NULL,Size,PROT_READ | PROT_WRITE,MAP_PRIVATE | MAP_ANONYMOUS,-1,0);
    int maxnodes=std::min(omp_get_max_threads(),numa_num_configured_nodes());
    size_t BytesPerNode=Size/maxnodes;
    char *cptr=(char *)ptr;
#ifdef TRACE_NUMA_ALLOC
    kma_printf("Memory allocation of %ld bytes, NUMA nodes: %d, BytesPernode: %ld\n",Size,maxnodes,BytesPerNode);
    kma_printf("mmap returned %lx\n",ptr);
#endif
    if (maxnodes<=1)
        return ptr;
    for(int i=0;i<maxnodes;i++) {
        void *mapptr=(void *)((unsigned long)cptr & 0xfffffffffffff000ul);
#ifdef TRACE_NUMA_ALLOC
        kma_printf("Block %d, cptr: %lx, mapptr: %lx\n",i,cptr,mapptr);
#endif
        numa_tonode_memory(mapptr ,BytesPerNode,i);
        cptr+=BytesPerNode;
    }
    return ptr;
}

void ExplicitNUMAAllocator::Free(void *ptr,size_t Size) {
    munmap(ptr,Size);
}

void ExplicitNUMAAllocator::PrintInfo() {
    kma_printf("Explicit NUMA allocator\n");
}

void *MallocNUMAAllocator::Alloc(size_t Size) {
    return malloc(Size);
}

void MallocNUMAAllocator::Free(void *ptr, size_t Size) {
    free(ptr);
}

void MallocNUMAAllocator::PrintInfo() {
    kma_printf("Malloc NUMA allocator\n");
}
