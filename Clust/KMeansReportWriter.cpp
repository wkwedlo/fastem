
 /* KMeansReportWriter.cpp
 *
 *  Created on: Apr 24, 2017
 *      Author: wkwedlo
 */
#include <limits>
#include <stdio.h>

#include "../Util/FileException.h"
#include "../Util/OpenMP.h"
#include "../Util/StdOut.h"

#include "KMeansReportWriter.h"


KMeansReportWriter::KMeansReportWriter(MultithreadedDataset &D,int ncl,const char *fstem) : CV(ncl,D.GetColCount()),Data(D) {
	ncols=CV.GetNCols();
	nclusters=ncl;
	filestem=fstem;
}


void KMeansReportWriter::QuantizeDataset(DynamicArray<OPTFLOAT> &v,const char *fname) {
	FILE *file=fopen(fname,"wb");
	if (file==NULL)
		throw FileException("Cannot open file for writing");
	long nRows=Data.GetRowCount();

    int minusOne=-1;
    fwrite(&minusOne,sizeof(minusOne),1,file);
	fwrite(&ncols,sizeof(ncols),1,file);
    fwrite(&nRows,sizeof(nRows),1,file);

	for(long i=0;i<Data.GetRowCount();i++) {
		const float *row=Data.GetRowNew(i);
		OPTFLOAT minssq=std::numeric_limits<OPTFLOAT>::max();
		int bestj=-1;
		for(int j=0;j<nclusters;j++) {
			OPTFLOAT ssq=CV.SquaredDistance(j,v,row);
			if (ssq<minssq) {
				minssq=ssq;
				bestj=j;
			}
		}
		fwrite(v.GetData()+bestj*ncols,sizeof(float),ncols,file);
	}
	fclose(file);
}

void KMeansReportWriter::DumpClusters(DynamicArray<OPTFLOAT> &v) {
    ASSERT(v.GetSize() == ncols*nclusters);
    kma_printf("Cluster centroids:\n");
    DynamicArray<DynamicArray<long> > Counts;

    Counts.SetSize(omp_get_max_threads());
#pragma omp parallel  shared(v, Counts)
      {
        int threadNum = omp_get_thread_num();
        Counts[threadNum].SetSize(nclusters);
        for (int i = 0; i < nclusters; i++) {
            Counts[threadNum][i] = 0;
        }

#pragma omp for
        for (int i = 0; i < Data.GetRowCount(); i++) {
            const float *row = Data.GetRowNew(i);
            OPTFLOAT minssq = std::numeric_limits<OPTFLOAT>::max();
            int bestj = -1;
            for (int j = 0; j < nclusters; j++) {
                OPTFLOAT ssq = CV.SquaredDistance(j, v, row);
                if (ssq < minssq) {
                    minssq = ssq;
                    bestj = j;
                }
            }
            Counts[threadNum][bestj]++;
        }
      }

    // Reduction
    DynamicArray<long> ReducedCounts(nclusters);
    for (int i = 0; i < nclusters; i++) {
        for (int j = 0; j < omp_get_max_threads(); j++) {
            ReducedCounts[i] += Counts[j][i];
        }
    }

    for(int i = 0; i < nclusters; i++) {
        for(int j = 0; j < ncols; j++) {
            kma_printf("%1.2f ", v[i*ncols+j]);
        }
        kma_printf(": %ld objects\n", ReducedCounts[i]);
    }
}

void KMeansReportWriter::DumpCentroids(DynamicArray<OPTFLOAT> &v) {
	for(int i=0;i<nclusters;i++) {
		for(int j=0;j<ncols;j++)
			kma_printf("%1.2f ",v[i*ncols+j]);
		kma_printf("\n");
	}
}

void KMeansReportWriter::WriteCentroids(DynamicArray<OPTFLOAT> &v,const char *fname) {
	FILE *file=fopen(fname,"wb");
	if (file==NULL)
		throw FileException("Cannot open centroid file for writing");
	fwrite(&nclusters,sizeof(nclusters),1,file);
	fwrite(&ncols,sizeof(ncols),1,file);
	fwrite(v.GetData(),sizeof(OPTFLOAT),ncols*nclusters,file);
	fclose(file);
}

void KMeansReportWriter::WriteClasses(DynamicArray<OPTFLOAT> &v,const char *fname) {
    FILE *cf = fopen(fname, "wb");
    if (cf == NULL)
        throw FileException("Cannot open class file for writing");
    long nRows = Data.GetRowCount();
    DynamicArray<int> ClNums(nRows);
    CV.ClassifyDataset(v, Data, ClNums);
    for (long i = 0; i < nRows; i++)
        ClNums[i] += 1;


    if (nRows > std::numeric_limits<int>::max()) {
        int minusOne = -1;
        fwrite(&minusOne, sizeof(minusOne), 1, cf);
        fwrite(&nRows, sizeof(nRows), 1, cf);
    } else {
        int smallnRows=(int)nRows;
        fwrite(&smallnRows, sizeof(smallnRows), 1, cf);
    }
	fwrite(ClNums.GetData(),sizeof(int),nRows,cf);
	fclose(cf);
}

void KMeansReportWriter::IterationReport(DynamicArray<OPTFLOAT> &v,int i) {
	const int BufferLen=512;
	char Buffer[BufferLen];
	snprintf(Buffer,BufferLen,"%s_%d.cls",filestem,i);
	WriteClasses(v,Buffer);
	snprintf(Buffer,BufferLen,"%s_%d.cnt",filestem,i);
	WriteCentroids(v,Buffer);
}

