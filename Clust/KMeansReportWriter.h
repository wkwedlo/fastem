/*
 * KMeansReportWriter.h
 *
 *  Created on: Apr 24, 2017
 *      Author: wkwedlo
 */

#ifndef CLUST_KMEANSREPORTWRITER_H_
#define CLUST_KMEANSREPORTWRITER_H_

#include "CentroidVector.h"
#include "../Util/MultithreadedDataset.h"

class KMeansReportWriter {
	int ncols;
	int nclusters;
	const char *filestem;

	const MultithreadedDataset &Data;
	CentroidVector CV;

public:
	KMeansReportWriter(MultithreadedDataset &D,int ncl,const char *fstem);
	void DumpClusters(DynamicArray<OPTFLOAT> &v);
	void DumpCentroids(DynamicArray<OPTFLOAT> &v);
	void QuantizeDataset(DynamicArray<OPTFLOAT> &v,const char *fname);
	void WriteCentroids(DynamicArray<OPTFLOAT> &v,const char *fname);
	void WriteClasses(DynamicArray<OPTFLOAT> &v,const char *fname);

	void IterationReport(DynamicArray<OPTFLOAT> &v,int i);

	virtual ~KMeansReportWriter() {}

};

#endif /* CLUST_KMEANSREPORTWRITER_H_ */
