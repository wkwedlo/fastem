/*
 * EigenUtil.h
 *
 *  Created on: Mar 6, 2017
 *      Author: wkwedlo
 */

#ifndef UTIL_EIGENUTIL_H_
#define UTIL_EIGENUTIL_H_

#include <stdio.h>

#include <Eigen/Dense>
#include "NumaAlloc.h"
#include "../Util/Optfloat.h"

typedef Eigen::Matrix<OPTFLOAT,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> EigenMatrix;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> EigenFloatMatrix;
typedef Eigen::Matrix<OPTFLOAT,Eigen::Dynamic,1> EigenVector;
typedef Eigen::Matrix<OPTFLOAT,1,Eigen::Dynamic> EigenRowVector;


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
