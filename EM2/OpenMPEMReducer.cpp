/*
 * OpenMPEMReducer.cpp
 *
 *  Created on: Apr 6, 2017
 *      Author: wkwedlo
 */

#include <stdexcept>
#include "../Util/StdOut.h"
#include "../Util/OpenMP.h"
#include "OpenMPEMReducer.h"



Step1ReducerData::Step1ReducerData(int K,int nCols) {
    PosteriorSum.resize(K);
    MeanSums.resize(K,nCols);
}

void Step1ReducerData::ClearData() {
    int K=MeanSums.rows();
    int nCols=MeanSums.cols();
    
    PosteriorSum=EigenRowVector::Zero(K);
    MeanSums=EigenMatrix::Zero(K,nCols);
}

Step1ReducerData &Step1ReducerData::operator+=(const Step1ReducerData &x) {
    int K=MeanSums.rows();
    int nCols=MeanSums.cols();
    ASSERT(K==x.MeanSums.rows());
    ASSERT(nCols==x.MeanSums.cols());
    PosteriorSum+=x.PosteriorSum;
    MeanSums+=x.MeanSums;
    return *this;
}

int Step1ReducerData::GetPackSize() {
    int K=MeanSums.rows();
    int nCols=MeanSums.cols();
    return K*(nCols+1);
}

void Step1ReducerData::PackData(DynamicArray<OPTFLOAT> &Buff) {
    int K=MeanSums.rows();
    int nCols=MeanSums.cols();
    int Cntr=0;
    for(int i=0;i<K;i++)
        Buff[Cntr++]=PosteriorSum[i];
    for(int i=0;i<K;i++)
        for(int j=0;j<nCols;j++)
            Buff[Cntr++]=MeanSums(i,j);
}

void Step1ReducerData::UnpackData(const DynamicArray<OPTFLOAT> &Buff) {
    int K=MeanSums.rows();
    int nCols=MeanSums.cols();
    int Cntr=0;
    for(int i=0;i<K;i++)
        PosteriorSum[i]=Buff[Cntr++];
    for(int i=0;i<K;i++)
        for(int j=0;j<nCols;j++)
            MeanSums(i,j)=Buff[Cntr++];
}


void Step2ReducerData::ClearData() {
    int K=CovSums.GetSize();
    int nCols=CovSums[0].cols();
    for(int i=0;i<K;i++) {
        CovSums[i]=EigenMatrix::Zero(nCols,nCols);
    }
}

Step2ReducerData & Step2ReducerData::operator+=(const Step2ReducerData &x) {
    int K=CovSums.GetSize();
    ASSERT(K==x.CovSums.GetSize());
    for(int i=0;i<K;i++) {
        CovSums[i]+=x.CovSums[i];
    }    
    return *this;
}

Step2ReducerData::Step2ReducerData(int K,int nCols) {
    CovSums.SetSize(K);
    for(int i=0;i<K;i++)
        CovSums[i].resize(nCols,nCols);
}

int Step2ReducerData::GetPackSize() {
    int K=CovSums.GetSize();
    int nCols=CovSums[0].cols();
    return K*nCols*(nCols+1)/2;
}

void Step2ReducerData::PackData(DynamicArray<OPTFLOAT> &Buff) {
    int K=CovSums.GetSize();
    int nCols=CovSums[0].cols();
    int Cntr=0;
    for (int k=0;k<K;k++) {
        const EigenMatrix &CovSum=CovSums[k];
        for(int i=0;i<nCols;i++)
            for(int j=i;j<nCols;j++)
                Buff[Cntr++]=CovSum(i,j);
    }
}

void Step2ReducerData::UnpackData(const DynamicArray<OPTFLOAT> &Buff) {
    int K=CovSums.GetSize();
    int nCols=CovSums[0].cols();
    int Cntr=0;
    for (int k=0;k<K;k++) {
        EigenMatrix &CovSum=CovSums[k];
        for(int i=0;i<nCols;i++)
            for(int j=i;j<nCols;j++)
                CovSum(i,j)=Buff[Cntr++];
    }
}
