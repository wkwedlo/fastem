/*
 * EigenUtil.h
 *
 *  Created on: Mar 6, 2017
 *      Author: wkwedlo
 */

#ifndef UTIL_EIGENUTIL_H_
#define UTIL_EIGENUTIL_H_

#include <stdio.h>
#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include <Eigen/Dense>
#include "NumaAlloc.h"
#include "../Util/Optfloat.h"

// #ifdef __AVX512F__
constexpr int kAlignBytes = 64;
constexpr auto kEigenAlignTag = Eigen::Aligned64;
// #else
// constexpr int kAlignBytes = 32;
// constexpr auto kEigenAlignTag = Eigen::Aligned32;
// #endif


using EigenMatrix = Eigen::Matrix<OPTFLOAT,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using EigenFloatMatrix= Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using EigenVector=Eigen::Matrix<OPTFLOAT,Eigen::Dynamic,1> ;
using EigenRowVector=Eigen::Matrix<OPTFLOAT,1,Eigen::Dynamic> ;
using EigenAlignedMap = Eigen::Map<EigenMatrix, kEigenAlignTag, Eigen::OuterStride<Eigen::Dynamic>>;


void print(const EigenRowVector &V, FILE *stream=stdout);
void print(const EigenMatrix &M,FILE *stream=stdout);

#ifdef EIGEN_RUNTIME_NO_MALLOC
void eigen_disable_malloc();
void eigen_enable_malloc();
#else
extern inline void eigen_disable_malloc() {}
extern inline void eigen_enable_malloc() {}
#endif

template <typename Matrix> double NumaLocalityofPrivateEigenMatrix (Matrix & M) {

	long nRows=M.rows();
	int DesiredCount=0;
	int CPU=sched_getcpu();
	int DesiredNode=numa_node_of_cpu(CPU);

	for(long i=0;i<nRows;i++) {
		void *ptr=M.row(i).data();
		int RealNode=NodeOfAddr(ptr);
		if (DesiredNode==RealNode)
			DesiredCount++;
	}

	return (double)DesiredCount/(double)nRows;
}

#endif /* UTIL_EIGENUTIL_H_ */
