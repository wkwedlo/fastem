/*
 * NormalDensity.h
 *
 *  Created on: Apr 23, 2010
 *      Author: wkwedlo
 */

#ifndef NORMALDENSITY_H_
#define NORMALDENSITY_H_

#include "../Util/MultithreadedDataset.h"
#include "../Util/EigenUtil.h"
#include "../Util/Optfloat.h"










class NormalDensity {
	int Dim;

	EigenMatrix C;
	EigenMatrix CholInvC;
	
	EigenRowVector M;
	OPTFLOAT Determinant;
	OPTFLOAT LogDeterminant;

	OPTFLOAT Coeff;
	OPTFLOAT LogCoeff;

protected:
	OPTFLOAT ComputeDot(const OPTFLOAT * __restrict__ row);
public:
	EigenRowVector GetMean() const {return M;}
	EigenMatrix GetCovariance() const {return C;}
	const EigenMatrix & GetInvertedCholCov() const {return CholInvC;}
	
	OPTFLOAT GetDeterminant() const {return Determinant;}
	OPTFLOAT GetLogDeterminant() const {return LogDeterminant;}
	int GetDimension() const {return Dim;}
	int ScanMatrix();
	void Init(const EigenMatrix &aC,const EigenRowVector &aM);
	NormalDensity(int aDim);
	OPTFLOAT GetDensity(const OPTFLOAT * __restrict__ row);
    OPTFLOAT GetDensity(const float * __restrict__ row);
    OPTFLOAT GetLogDensity(const OPTFLOAT * __restrict__ row);
	OPTFLOAT MahalanobisDist(const EigenRowVector &row);
	OPTFLOAT ComputeTrace();
	OPTFLOAT ComputeConditionNumber();
	virtual ~NormalDensity();
};


class BhattDist {
protected:
	NormalDensity Dens;
	EigenMatrix Sum;
public:
	BhattDist(int aDim);
	OPTFLOAT ComputeDist(const NormalDensity &Dens1,const NormalDensity &Dens2);


};
#endif /* NORMALDENSITY_H_ */
