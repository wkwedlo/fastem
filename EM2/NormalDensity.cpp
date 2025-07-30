/*
 * NormalDensity.cpp
 *
 *  Created on: Apr 23, 2010
 *      Author: wkwedlo
 */
#include <cmath>
#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include "NormalDensity.h"
#include "EMException.h"

#include "../Util/Debug.h"
#include "../Util/EigenUtil.h"






NormalDensity::NormalDensity(int aDim) {

	Dim=aDim;
	Determinant=0.0;
	Coeff=0.0;
	LogCoeff=0.0;
	C.resize(Dim,Dim);
	CholInvC.resize(Dim,Dim);
	M.resize(Dim);

}

NormalDensity::~NormalDensity() {
}



void NormalDensity::Init(const EigenMatrix &aC,const EigenRowVector &aM) {
	M=aM;
	C=aC.selfadjointView<Eigen::Lower>();

	Eigen::LLT<EigenMatrix> LLT(C);
	if (LLT.info()!=Eigen::Success)
		throw EMException("Cholesky decomposition failed");

	EigenMatrix L=LLT.matrixL();

	LogDeterminant=2.0*L.diagonal().array().log().sum();
	Determinant=std::exp(LogDeterminant);


	LogCoeff=-0.5*Dim*log(2.0*M_PI)-0.5*LogDeterminant;
	Coeff=std::exp(LogCoeff);
	//TRACE4("Det: %g LogDet %g Coeff: %g LogCoeff: %g\n",Determinant,LogDet,Coeff,LogCoeff);

	CholInvC=EigenMatrix::Identity(Dim,Dim);
	L.triangularView<Eigen::Lower>().solveInPlace(CholInvC);
	CholInvC.transposeInPlace();

}

OPTFLOAT NormalDensity::ComputeConditionNumber() {
	Eigen::SelfAdjointEigenSolver<EigenMatrix> Solver(C,Eigen::EigenvaluesOnly);
	return std::fabs(Solver.eigenvalues()(Dim-1)/Solver.eigenvalues()(0));

}

OPTFLOAT NormalDensity::ComputeTrace() {
	OPTFLOAT Sum=0.0;
	for(int i=0;i<Dim;i++)
		Sum+=C(i,i);
	return Sum;
}

OPTFLOAT NormalDensity::ComputeDot(const OPTFLOAT * __restrict__ row) {
	OPTFLOAT * __restrict__ pWork1=(OPTFLOAT *)alloca(Dim*sizeof(OPTFLOAT));

#pragma omp simd
	for(int i=0;i<Dim;i++)
		pWork1[i]=row[i]-M[i];

	cblas_dtrmv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit,
				Dim, CholInvC.data(), Dim, pWork1, 1);

	OPTFLOAT Dot=0.5*cblas_ddot(Dim,pWork1,1,pWork1,1);
	ASSERT(Dot>=0.0);
	return Dot;
}

OPTFLOAT NormalDensity::MahalanobisDist(const EigenRowVector &row) {
	return 2.0*ComputeDot(row.data());
}

OPTFLOAT NormalDensity::GetDensity(const OPTFLOAT * __restrict__ row) {
	OPTFLOAT Dot=ComputeDot(row);
	OPTFLOAT Res=Coeff*std::exp(-(OPTFLOAT)Dot);
	return Res;

}

OPTFLOAT NormalDensity::GetDensity(const float * __restrict__ row) {
    OPTFLOAT drow[Dim];
    for(int i=0;i<Dim;i++)
        drow[i]=row[i];
    OPTFLOAT Dot=ComputeDot(drow);
    OPTFLOAT Res=Coeff*std::exp(-(OPTFLOAT)Dot);
    return Res;

}


OPTFLOAT NormalDensity::GetLogDensity(const OPTFLOAT * __restrict__ row) {
	OPTFLOAT Dot=ComputeDot(row);
	return LogCoeff-Dot;
}



BhattDist::BhattDist(int aDim) : Dens(aDim),Sum(aDim,aDim) {

}

OPTFLOAT BhattDist::ComputeDist(const NormalDensity &Dens1,const NormalDensity &Dens2) {
	Sum=0.5*(Dens1.GetCovariance()+Dens2.GetCovariance());
	Dens.Init(Sum,Dens1.GetMean());
	OPTFLOAT D1=0.125*Dens.MahalanobisDist(Dens2.GetMean());
	OPTFLOAT D2=0.5*log(Dens.GetDeterminant()/sqrt(Dens1.GetDeterminant()*Dens2.GetDeterminant()));
	OPTFLOAT D=D1+D2;
	if (D<0.0)
		D=0.0;
	ASSERT(D>=0.0);
	return D;
}

