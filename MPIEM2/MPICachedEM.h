//
// Created by wkwedlo on 05.04.23.
//

#ifndef CLUST_MPICACHEDEM_H
#define CLUST_MPICACHEDEM_H


#include "../EM2/CachedEM.h"
#include "MPIUniversalReducer.h"


class MPICachedEM : public CachedEM {
        MPIUniversalReducer<Step1ReducerData> *pMPIReducer1;
        MPIUniversalReducer<Step2ReducerData> *pMPIReducer2;

protected:
    virtual void MPIReduceStep1Data();
    virtual void MPIReduceStep2Data();
    virtual void MPIReduceLogL(OPTFLOAT &aLogl);

public:
    virtual const char *GetName() const {return "MPI CachedEM";}
    MPICachedEM(int ncols,int nrows,int ncl,const char *ReducerName,int anDensBlocks=1,int anMStepBlocks=1);
    virtual ~MPICachedEM();


};


#endif //CLUST_MPICACHEDEM_H
