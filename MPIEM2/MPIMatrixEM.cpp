//
// Created by wkwedlo on 04.08.23.
//

#include "MPIMatrixEM.h"

//void MPIMatrixEM::PrintNumaLocalityInfo() {
//    int nThreads=omp_get_max_threads();
//    double LocalitySumTD=0.0,LocalitySumM2=0.0;
//
//#pragma omp parallel reduction(+ : LocalitySumTD,LocalitySumM2)
//    {
//        int t=omp_get_thread_num();
//        LocalitySumTD=NumaLocalityofPrivateEigenMatrix(ThreadInfo[t].TempDataset);
//        LocalitySumM2=NumaLocalityofPrivateEigenMatrix(ThreadInfo[t].M2);
//    }
//    LocalitySumM2/=(double)nThreads;
//    LocalitySumTD/=(double)nThreads;
//    double LocalityPosterior=NumaLocalityofLargeArray(Posteriors,BM.GetBlockCount());
//
//    double MPIM2Locality=0.0,MPITDLocality=0.0,MPIPostLocality=0.0;
//    int nProcs;
//    MPI_Comm_size(MPI_COMM_WORLD,&nProcs);
//    MPI_Reduce(&LocalitySumTD,&MPITDLocality,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
//    MPI_Reduce(&LocalitySumM2,&MPIM2Locality,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
//    MPI_Reduce(&LocalityPosterior,&MPIPostLocality,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
//
//
//
//    kma_printf("Average NUMA locality of thread private  MatrixEM::TempDataset: %1.3f%%\n",100.0*MPITDLocality/(double)nProcs);
//    kma_printf("Average NUMA locality of thread private MatrixEM::M2: %1.3f%%\n",100.0*MPIM2Locality/(double)nProcs);
//    pMStep->PrintNumaLocalityInfo();
//    kma_printf("NUMA locality of posterior matrix : %1.3f%%\n",100.0*MPIPostLocality/(double)nProcs);
//
//}


MPIMatrixEM::MPIMatrixEM(int ncols, int nrows, int ncl, const char *ReducerName,int anDensBlocks,int anMStepBlocks) :
MatrixEM( ncols, nrows, ncl, ReducerName,anDensBlocks,anMStepBlocks)
{
    Step1ReducerData x(ncl,ncols);
    pMPIReducer1=new MPIUniversalReducer<Step1ReducerData>(x);
    Step2ReducerData y(ncl,ncols);
    pMPIReducer2=new MPIUniversalReducer<Step2ReducerData>(y);
}

MPIMatrixEM::~MPIMatrixEM() {
    delete pMPIReducer1;
    delete pMPIReducer2;
}


void MPIMatrixEM::MPIReduceLogL(OPTFLOAT &aLogL) {
    const OPTFLOAT SendBuffer=aLogL;
    MPI_Allreduce(&SendBuffer,&aLogL,1,MPI_OPTFLOAT,MPI_SUM,MPI_COMM_WORLD);
}

void MPIMatrixEM::MPIReduceStep1Data() {
    pMPIReducer1->AllReduce(pReducer1->GetData(0));
}

void MPIMatrixEM::MPIReduceStep2Data() {
    pMPIReducer2->AllReduce(pReducer2->GetData(0));
}




SimpleMPIMatrixEM::SimpleMPIMatrixEM(int ncols, int nrows, int ncl,const char *ReducerName, int anDensBlocks,int anMStepBlocks) :
    SimpleMatrixEM(ncols, nrows, ncl, ReducerName, anDensBlocks, anMStepBlocks)
{
    Step1ReducerData x(ncl,ncols);
    pMPIReducer1=new MPIUniversalReducer<Step1ReducerData>(x);
    Step2ReducerData y(ncl,ncols);
    pMPIReducer2=new MPIUniversalReducer<Step2ReducerData>(y);
}

void SimpleMPIMatrixEM::MPIReduceLogL(OPTFLOAT &aLogL) {
    const OPTFLOAT SendBuffer=aLogL;
    MPI_Allreduce(&SendBuffer,&aLogL,1,MPI_OPTFLOAT,MPI_SUM,MPI_COMM_WORLD);
}

void SimpleMPIMatrixEM::MPIReduceStep1Data() {
    pMPIReducer1->AllReduce(pReducer1->GetData(0));
}

void SimpleMPIMatrixEM::MPIReduceStep2Data() {
    pMPIReducer2->AllReduce(pReducer2->GetData(0));
}

SimpleMPIMatrixEM::~SimpleMPIMatrixEM() {
    delete pMPIReducer1;
    delete pMPIReducer2;
}
