/*
 * OpenMPEMReducer.h
 *
 *  Created on: Apr 6, 2017
 *      Author: wkwedlo
 */

#ifndef EM2_OPENMPEMREDUCER_H_
#define EM2_OPENMPEMREDUCER_H_

#include "../Util/EigenUtil.h"
#include "../Util/Array.h"
#include "../Util/OMPReducer.h"


struct Step1ReducerData {
		EigenRowVector PosteriorSum;
		EigenMatrix MeanSums;
		
		void ClearData();
		Step1ReducerData &operator+=(const Step1ReducerData &x);
		Step1ReducerData(int K,int nCols);
		Step1ReducerData() {}
        int GetPackSize();
        void PackData(DynamicArray<OPTFLOAT> &Buff);
        void UnpackData(const DynamicArray<OPTFLOAT> &Buff);
};

struct Step2ReducerData {
		DynamicArray<EigenMatrix> CovSums;
		void ClearData();
		Step2ReducerData &operator+=(const Step2ReducerData &x);
		Step2ReducerData(int K,int nCols);
		Step2ReducerData() {}

        int GetPackSize();
        void PackData(DynamicArray<OPTFLOAT> &Buff);
        void UnpackData(const DynamicArray<OPTFLOAT> &Buff);
};




#endif /* EM2_OPENMPEMREDUCER_H_ */
