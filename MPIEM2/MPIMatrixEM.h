//
// Created by wkwedlo on 04.08.23.
//

#ifndef CLUST_MPIMATRIXEM_H
#define CLUST_MPIMATRIXEM_H

#include "../EM2/MatrixEM.h"
#include "MPIUniversalReducer.h"


class MPIMatrixEM : public MatrixEM {
protected:
    virtual void MPIReduceLogL(OPTFLOAT &aLogl);
    MPIUniversalReducer<Step1ReducerData> *pMPIReducer1;
    MPIUniversalReducer<Step2ReducerData> *pMPIReducer2;

    virtual void MPIReduceStep1Data();
    virtual void MPIReduceStep2Data();
public:
    virtual const char *GetName() const {return "MPI MatrixEM";}
    MPIMatrixEM(int ncols, int nrows, int ncl, const char *ReducerName,int anDensBlocks=1,int anMStepBlocks=1);
    virtual ~MPIMatrixEM();
};

class SimpleMPIMatrixEM : public SimpleMatrixEM {
protected:
    virtual void MPIReduceLogL(OPTFLOAT &aLogl);
    MPIUniversalReducer<Step1ReducerData> *pMPIReducer1;
    MPIUniversalReducer<Step2ReducerData> *pMPIReducer2;
    virtual void MPIReduceStep1Data();
    virtual void MPIReduceStep2Data();
public:
    virtual const char *GetName() const {return "Simple MPI MatrixEM";}
    SimpleMPIMatrixEM(int ncols, int nrows, int ncl, const char *ReducerName,int anDensBlocks=1,int anMStepBlocks=1);
    virtual ~SimpleMPIMatrixEM();
};

#endif //CLUST_MPIMATRIXEM_H
