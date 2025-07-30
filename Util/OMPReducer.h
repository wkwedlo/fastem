#ifndef UTIL_OMPREDUCER_H_
#define UTIL_OMPREDUCER_H_

#include "Array.h"
#include "OpenMP.h"
#include "StdOut.h"

template<typename T> class OMPReducer {
protected:
    DynamicArray<T> ThreadData;
public:
    OMPReducer(const T &val);
    void SetData(int tid,const T &Data) {ThreadData[tid]=Data;}
    T &GetData(int tid) {return ThreadData[tid];}
    void ClearAllData();
    virtual void ReduceToFirstThread()=0;
    virtual void PrintInfo()=0;
    virtual ~OMPReducer() {;}
};

template <typename T> OMPReducer<T>::OMPReducer(const T &val) {
    int nThreads=omp_get_max_threads();
    ThreadData.SetSize(nThreads);
#pragma omp parallel default(none) shared(val)
    {
        int tid=omp_get_thread_num();
        ThreadData[tid]=val;
    }    
}


template <typename T> void OMPReducer<T>::ClearAllData() {
#pragma omp parallel default(none)
    {
        int tid=omp_get_thread_num();
        ThreadData[tid].ClearData();
    }    
}

template<typename T> class NaiveOMPReducer : public OMPReducer<T> {
    using OMPReducer<T>::ThreadData;
public:
    virtual void ReduceToFirstThread();
    virtual void PrintInfo() {kma_printf("Naive OpenMP reducer\n");}
    NaiveOMPReducer(const T &val) : OMPReducer<T>(val) {}

};



template<typename T> void NaiveOMPReducer<T>::ReduceToFirstThread() {
	int nThreads=ThreadData.GetSize();
    for (int tid=1;tid<nThreads;tid++) {
        ThreadData[0]+=ThreadData[tid];
	}
}


template<typename T> class Log2OMPReducer : public OMPReducer<T> {
    using OMPReducer<T>::ThreadData;

public:
    virtual void ReduceToFirstThread();
    virtual void PrintInfo() {kma_printf("Log2 OpenMP reducer\n");}
    Log2OMPReducer(const T &val) : OMPReducer<T>(val) {}
};


template<typename T> void Log2OMPReducer<T>::ReduceToFirstThread() {
	int nThreads=ThreadData.GetSize();
#pragma omp parallel
	{
		int tid=omp_get_thread_num();
		for(int s=1; s<nThreads;s*=2) {
			if (tid % (2*s)==0 && tid+s<nThreads)
				ThreadData[tid]+=ThreadData[tid+s];
#pragma omp barrier
		}
	}
}

#endif