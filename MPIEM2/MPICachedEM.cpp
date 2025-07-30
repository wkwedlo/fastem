#include <mpi.h>

#include "../Util/Debug.h"
#include "MPICachedEM.h"


MPICachedEM::MPICachedEM(int ncols, int nrows, int ncl,const char * ReducerName,int anDensBlocks,int anMStepBlocks) :
                CachedEM(ncols, nrows, ncl,ReducerName,anDensBlocks,anMStepBlocks) {
    Step1ReducerData x(ncl,ncols);
    pMPIReducer1=new MPIUniversalReducer<Step1ReducerData>(x);
    Step2ReducerData y(ncl,ncols);
    pMPIReducer2=new MPIUniversalReducer<Step2ReducerData>(y);
}

void MPICachedEM::MPIReduceStep1Data() {
    pMPIReducer1->AllReduce(pReducer1->GetData(0));
}

void MPICachedEM::MPIReduceStep2Data() {
    pMPIReducer2->AllReduce(pReducer2->GetData(0));
}

void MPICachedEM::MPIReduceLogL(OPTFLOAT &aLogL) {
    const OPTFLOAT SendBuffer=aLogL;
    MPI_Allreduce(&SendBuffer,&aLogL,1,MPI_OPTFLOAT,MPI_SUM,MPI_COMM_WORLD);
}

MPICachedEM::~MPICachedEM() {
    delete pMPIReducer1;
    delete pMPIReducer2;
}

