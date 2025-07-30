//
// Created by wkwedlo on 04.08.23.
//
//

#ifndef CLUST_UNIVERSALREDUCER_H
#define CLUST_UNIVERSALREDUCER_H

#include <mpi.h>

#include "../Util/Optfloat.h"
#include "../Util/Array.h"

template <typename  T> class MPIUniversalReducer {

protected:
    DynamicArray<OPTFLOAT> SendBuffer;
    DynamicArray<OPTFLOAT> RecvBuffer;

public:
    MPIUniversalReducer(T x);
    void AllReduce(T &x);
};

template <typename T> MPIUniversalReducer<T>::MPIUniversalReducer(T x) {
    int Size=x.GetPackSize();
    SendBuffer.SetSize(Size);
    RecvBuffer.SetSize(Size);
}

template<typename T> void MPIUniversalReducer<T>::AllReduce(T &x) {
    x.PackData(SendBuffer);
    MPI_Allreduce(SendBuffer.GetData(),RecvBuffer.GetData(),RecvBuffer.GetSize(),MPI_OPTFLOAT,MPI_SUM,MPI_COMM_WORLD);
    x.UnpackData(RecvBuffer);
}




#endif //CLUST_MPICACHEDEMREDUCER_H
