/*
 * CPUAffinityInfo.h
 *
 *  Created on: Feb 15, 2018
 *      Author: wkwedlo
 */

#ifndef UTIL_MPITHREADAFFINITYINFO_H_
#define UTIL_MPITHREADAFFINITYINFO_H_

#include <stdio.h>
#include "ThreadAffinityInfo.h"

class MPIThreadAffinityInfo : public ThreadAffinityInfo {
protected:
	int MPIRank,MPISize;

	DynamicArray<int> DstCPUs;
	DynamicArray<int> DstNodes;

	void MergeMPIInfos();
	void PrintGlobalArray(FILE *pF,const DynamicArray<int> &DstArr);
public:
	MPIThreadAffinityInfo();
	virtual void PrintReport(const char *fname);
	virtual ~MPIThreadAffinityInfo() {}

};

#endif /* UTIL_THREADAFFINITYINFO_H_ */
