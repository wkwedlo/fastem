/*
 * EigenUtil.cpp
 *
 *  Created on: Mar 7, 2017
 *      Author: wkwedlo
 */

#include "EigenUtil.h"


#ifdef EIGEN_RUNTIME_NO_MALLOC
void eigen_disable_malloc() {
	Eigen::internal::set_is_malloc_allowed(false);
}
void eigen_enable_malloc() {
	Eigen::internal::set_is_malloc_allowed(true);
}
#endif



void print(const EigenRowVector &V,FILE *stream) {
	for(long i=0;i<V.size();i++)
		fprintf(stream,"%.3g ",V[i]);
	fprintf(stream,"\n");
}

void print(const EigenMatrix &M,FILE *stream) {
	for(long i=0;i<M.rows();i++) {
		for(long j=0;j<M.cols();j++)
			fprintf(stream,"%.3g ",M(i,j));
		fprintf(stream,"\n");
	}
}


