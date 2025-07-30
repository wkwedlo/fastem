#ifndef OPENMP_H_
#define OPENMP_H_

#include "Debug.h"



#ifdef _OPENMP
#include <omp.h>
#else
extern inline int omp_get_num_threads() {
	return 1;
}

extern inline int omp_get_thread_num() {
	return 0;
}

extern inline int omp_get_max_threads() {
	return 1;
}

extern inline int omp_get_active_level() {
	return 0;
}

extern inline int omp_get_max_active_levels() {
	return 1;
}


extern inline int omp_get_ancestor_thread_num(int level) {
    return 0;
}
#endif

template<typename T> extern inline void FindOpenMPItems(T nItems,T &Offset,T &Count,int Size,int Id) {
    Count=nItems/Size;
    int SubRemainder=nItems % Size;
    if (Id<SubRemainder) {
        Count++;
        Offset=Count*Id;
    } else {
        Offset=Count*Id+SubRemainder;
    }
}


template<typename T> extern inline void FindOpenMPItems(T nItems,T &Offset,T &Count) {
    int Size=omp_get_num_threads();
    int Id=omp_get_thread_num();
    FindOpenMPItems(nItems,Offset,Count,Size,Id);
}



#ifndef OMPDYNAMIC
#define OMPDYNAMIC schedule(static)
#endif

#endif /* OPENMP_H_ */
