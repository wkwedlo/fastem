//
// Created by wkwedlo on 04.08.23.
//

#include "MPIMatrixEM.h"




MPIMatrixEM::MPIMatrixEM(int ncols, int nrows, int ncl, const char *ReducerName,int anDensBlocks,int anMStepBlocks) :
MatrixEM( ncols, nrows, ncl, ReducerName,anDensBlocks,anMStepBlocks)
{
    Step1ReducerData x(ncl,ncols);
    pMPIReducer1=new MPIUniversalReducer<Step1ReducerData>(x);
}

MPIMatrixEM::~MPIMatrixEM() {
    delete pMPIReducer1;
}


void MPIMatrixEM::MPIReduceLogL(OPTFLOAT &aLogL) {
    const OPTFLOAT SendBuffer=aLogL;
    MPI_Allreduce(&SendBuffer,&aLogL,1,MPI_OPTFLOAT,MPI_SUM,MPI_COMM_WORLD);
}

void MPIMatrixEM::MPIReduceStep1Data() {
    pMPIReducer1->AllReduce(pReducer1->GetData(0));
}

void MPIMatrixEM::MPIReduceStep2Data() {
    MPI_Allreduce(MPI_IN_PLACE,Step2CovSums[0].GetData(),Step2CovSums[0].GetPackedSize(),MPI_OPTFLOAT,MPI_SUM,MPI_COMM_WORLD);
}




SimpleMPIMatrixEM::SimpleMPIMatrixEM(int ncols, int nrows, int ncl,const char *ReducerName, int anDensBlocks,int anMStepBlocks) :
    SimpleMatrixEM(ncols, nrows, ncl, ReducerName, anDensBlocks, anMStepBlocks)
{
    Step1ReducerData x(ncl,ncols);
    pMPIReducer1=new MPIUniversalReducer<Step1ReducerData>(x);
}

void SimpleMPIMatrixEM::MPIReduceLogL(OPTFLOAT &aLogL) {
    const OPTFLOAT SendBuffer=aLogL;
    MPI_Allreduce(&SendBuffer,&aLogL,1,MPI_OPTFLOAT,MPI_SUM,MPI_COMM_WORLD);
}

void SimpleMPIMatrixEM::MPIReduceStep1Data() {
    pMPIReducer1->AllReduce(pReducer1->GetData(0));
}

void SimpleMPIMatrixEM::MPIReduceStep2Data() {
    MPI_Allreduce(MPI_IN_PLACE,Step2CovSums[0].GetData(),Step2CovSums[0].GetPackedSize(),MPI_OPTFLOAT,MPI_SUM,MPI_COMM_WORLD);
}

SimpleMPIMatrixEM::~SimpleMPIMatrixEM() {
    delete pMPIReducer1;
}
