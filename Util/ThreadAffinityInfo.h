/*
 * ThreadAffinityInfo.h
 *
 *  Created on: Feb 19, 2018
 *      Author: wkwedlo
 */

#ifndef UTIL_THREADAFFINITYINFO_H_
#define UTIL_THREADAFFINITYINFO_H_
#include "Array.h"

class ThreadAffinityInfo {
protected:
	int NThreads;
	DynamicArray<int> CPUs;
	DynamicArray<int> Nodes;
	void GetProcessInfo();
	void PrintArray(FILE *pF,const DynamicArray<int> &Arr);

public:
	virtual void PrintReport(const char *fname);
	ThreadAffinityInfo();
	virtual ~ThreadAffinityInfo();
};

#endif /* UTIL_THREADAFFINITYINFO_H_ */
