/*
 * MultithreadedDataset.cpp
 *
 *  Created on: Mar 28, 2013
 *      Author: wkwedlo
 */
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <numa.h>
#include <sched.h>
#include <limits>

#include "OpenMP.h"
#include "MultithreadedDataset.h"
#include "NumaAlloc.h"
#include "FileException.h"
#include "../EM2/BlockManager.h"

MultithreadedDataset::MultithreadedDataset(int anBlocks)  {
	nTotalRows=0;
	nBlocks=anBlocks;
}


void MultithreadedDataset::LoadFromCenterArray(DynamicArray< DynamicArray<float> > &Centers) {
	int nCols=Centers[0].GetSize();
	long nRows=nTotalRows=Centers.GetSize();


	Data.SetSize(nRows,nCols);
	long nStride=Data.GetStride();
	float *pData=Data.GetData();

	for(long i=0;i<nRows;i++) {
			DynamicArray<float> Center(Centers[i]);
			for(int j=0;j<nCols;j++)
				pData[j]=(float)Center[j];
			pData+=nStride;
	}
	ComputeStatistics();
}

void MultithreadedDataset::LoadFromCentroidVector(DynamicArray<OPTFLOAT> &vec,int ncols,int nclusters) {
	TraceCov=-1.0;


	Data.SetSize(nclusters,ncols);
	nTotalRows=nclusters;
	long nStride=Data.GetStride();

	for(int i=0;i<nclusters;i++) {
			for(int j=0;j<ncols;j++)
				(*this)(i,j)=(float)vec[i*ncols+j];
	}
	ComputeStatistics();
}




void MultithreadedDataset::LoadSelectedObjects(const MultithreadedDataset &Src,const DynamicArray<long> &ObjNums) {
	TraceCov=-1.0;
	
	int nCols=Src.GetColCount();
	
	long nRows=nTotalRows=ObjNums.GetSize();
	Data.SetSize(nRows,nCols);

	for(long i=0;i<nRows;i++) {
			for(int j=0;j<nCols;j++)
				(*this)(i,j)=Src(i,j);
	}
	ComputeStatistics();
}

void MultithreadedDataset::SwapRows(long i,long j) {
	int nCols=GetColCount();
	float Buffer[nCols];

	float * __restrict__ iptr=GetRowNew(i);
	float * __restrict__ jptr=GetRowNew(j);

	memcpy(Buffer,iptr,nCols*sizeof(float));
	memcpy(iptr,jptr,nCols*sizeof(float));
	memcpy(jptr,Buffer,nCols*sizeof(float));
}




MultithreadedDataset::~MultithreadedDataset()
{
}



void MultithreadedDataset::ReadChunks(int fd,float *pData,long ChunkCount,long FirstRow,DynamicArray<struct iovec> &IOVecs) {
	ASSERT(ChunkCount<=IOVecs.GetSize());
	int nCols=GetColCount();
	long nStride=GetStride();

	for(long i=0;i<ChunkCount;i++) {
		IOVecs[i].iov_base=pData;
		IOVecs[i].iov_len=nCols*sizeof(float);;
		pData+=nStride;

	}
	long BytesRead=readv(fd,IOVecs.GetData(),ChunkCount);
	long ExpectedRead=(long)ChunkCount*(long)nCols*sizeof(float);
	if(BytesRead!=ExpectedRead)
		throw FileException("Invalid number of bytes read by readv syscall");
}


void MultithreadedDataset::LoadHeader(int fd,long &nRows,int &nCols) {

    int nRowsSmall;
    if( read(fd,&nRowsSmall,sizeof(nRowsSmall))!=sizeof(nRowsSmall)
	   || read(fd,&nCols,sizeof(nCols))!=sizeof(nCols))
		  throw FileException("read failed");
    if (nRowsSmall==-1) {
        if (read(fd, &nRows, sizeof(nRows)) != sizeof(nRows))
            throw FileException("read failed");
        HeaderSize=sizeof(long)+2*sizeof(int);
    } else {
        nRows = nRowsSmall;
        HeaderSize = 2 * sizeof(int);
    }
    TRACE2("Total rows: %ld, Total cols: %d\n",nRows,nCols);
}



void MultithreadedDataset::PartialLoad(char *fname,long StartRow,long nR,int nC,long _Alignment) {


	Data.SetSize(nR,nC,_Alignment,false);
	long nStride=Data.GetStride();



	float *pBuff=Data.GetData();

    BlockManager BM(nR,nBlocks);

#pragma omp parallel default(none) shared(BM,stderr) firstprivate(StartRow,fname,nC,nR,pBuff,nStride)
	{


			int fdnew=open(fname,O_RDONLY);
			DynamicArray<struct iovec> IOVecs(1024);
			for (int k=0;k<nBlocks;k++) {
				long BlockStart,BlockRows;
				BM.GetBlock(k,BlockStart,BlockRows);
				long Offset,Count;
				BM.GetThreadSubblock(BlockStart,BlockRows,Offset,Count);
				VERIFY(lseek64(fdnew,HeaderSize+sizeof(float)*((long)StartRow+(long)Offset+(long)BlockStart)*(long)nC,SEEK_SET)>0L);
				float *ptr=pBuff+((long)Offset+(long)BlockStart)*(long)nStride;

				//TRACE4("Thread %d fd: %d tRows %ld mini %ld\n",id,fdnew,tRows,mini);

				while(Count>0) {
					int nChunks= Count>IOVecs.GetSize() ? (int)IOVecs.GetSize() : (int)Count;
					ReadChunks(fdnew,ptr,nChunks,Offset,IOVecs);
					ptr+=((long)nChunks*(long)nStride);
					Count-=nChunks;
					Offset+=nChunks;
				}
			}
			close(fdnew);
	}
	//FindMinMax();
}

void MultithreadedDataset::ComputeStatistics() {
	long nRows=GetRowCount();
	int nCols=GetColCount();
	int nThreads=omp_get_max_threads();


	OrigMeans.SetSize(nCols);
	OrigStdDevs.SetSize(nCols);
	Means.SetSize(nCols);
	StdDevs.SetSize(nCols);
	Max.SetSize(nCols);
	Min.SetSize(nCols);
	for(long i=0;i<nCols;i++) {
		OrigMeans[i]=Means[i]=OrigStdDevs[i]=0.0f;
		StdDevs[i]=1.0f;
		Max[i]=-std::numeric_limits<float>::max();
		Min[i]=std::numeric_limits<float>::max();
	}

	ThreadData.SetSize(nThreads);
	BlockManager BM(nRows,nBlocks);

#pragma omp parallel default(none) firstprivate(nRows,nCols) shared(BM)
	{
		int tid=omp_get_thread_num();
		DynamicArray<OPTFLOAT> &tMeanSum=ThreadData[tid].MeanSum;
		DynamicArray<float> &tMin=ThreadData[tid].Min;
		DynamicArray<float> &tMax=ThreadData[tid].Max;
		tMeanSum.SetSize(nCols);
		tMin.SetSize(nCols);
		tMax.SetSize(nCols);
		for (int j=0;j<nCols;j++) {
			tMeanSum[j]=0.0;
			tMin[j]=std::numeric_limits<float>::max();
			tMax[j]=-std::numeric_limits<float>::max();
		}

		for (int k=0;k<nBlocks;k++) {
			long BlockStart,BlockRows;
			BM.GetBlock(k,BlockStart,BlockRows);
			long Offset,Count;
			BM.GetThreadSubblock(BlockStart,BlockRows,Offset,Count);
			for(long i=BlockStart+Offset;i<BlockStart+Offset+Count;i++) {
				const float *rData=GetRowNew(i);
				for(int j=0;j<nCols;j++) {
					tMeanSum[j]+=rData[j];
					if (rData[j]>tMax[j])
						tMax[j]=rData[j];
					if (rData[j]<tMin[j])
						tMin[j]=rData[j];
				}

			}
		}
	}
	for (int tid=0;tid<nThreads;tid++) {
		for (int j=0;j<nCols;j++) {
			OrigMeans[j]+=ThreadData[tid].MeanSum[j];
			if (ThreadData[tid].Max[j]>Max[j])
				Max[j]=ThreadData[tid].Max[j];
			if (ThreadData[tid].Min[j]<Min[j])
				Min[j]=ThreadData[tid].Min[j];
		}
	}
	ReduceStatisticsStep1(OrigMeans,Min,Max);
	for(int j=0;j<nCols;j++)
		OrigMeans[j]/=(OPTFLOAT)nTotalRows;


#pragma omp parallel default(none) firstprivate(nRows,nCols) shared(BM)
	{
		int tid=omp_get_thread_num();
		DynamicArray<OPTFLOAT> &tStdDevSum=ThreadData[tid].StdDevSum;
		tStdDevSum.SetSize(nCols);
		for (int j=0;j<nCols;j++) {
			tStdDevSum[j]=0.0;
		}

		for (int k=0;k<nBlocks;k++) {
			long BlockStart,BlockRows;
			BM.GetBlock(k,BlockStart,BlockRows);
			long Offset,Count;
			BM.GetThreadSubblock(BlockStart,BlockRows,Offset,Count);
			for(long i=BlockStart+Offset;i<BlockStart+Offset+Count;i++) {
				const float *rData=GetRowNew(i);
				for(int j=0;j<nCols;j++) {
					tStdDevSum[j]+=((rData[j]-OrigMeans[j])*(rData[j]-OrigMeans[j]));
				}

			}
		}
	}
	for (int tid=0;tid<nThreads;tid++) {
		for (int j=0;j<nCols;j++) {
			OrigStdDevs[j]+=ThreadData[tid].StdDevSum[j];
		}
	}
	ReduceStatisticsStep2(OrigStdDevs);
	TraceCov=0.0;
	for(int j=0;j<nCols;j++) {
		OrigStdDevs[j]/=((OPTFLOAT)nTotalRows-1.0);
		TraceCov+=OrigStdDevs[j];
		OrigStdDevs[j]=std::sqrt(OrigStdDevs[j]);
		TRACE1("Column%d: ",j);
		TRACE4("mean: %5.6f std: %5.6f min: %5.6f max: %5.6f\n",OrigMeans[j],OrigStdDevs[j],Min[j],Max[j]);
	}
}



void MultithreadedDataset::Load(char *fname,long _Alignment) {
	int fd=open(fname,O_RDONLY);
	if (fd<0)
		throw FileException("Cannot open dataset");

	long nRows;
    int nCols;
	LoadHeader(fd,nRows,nCols);
	nTotalRows=nRows;
	close(fd);
	PartialLoad(fname,0,nRows,nCols,_Alignment);
	ComputeStatistics();
}

void MultithreadedDataset::QueryHeader(char *fname, long &nRows, int &nCols) {
	int fd=open(fname,O_RDONLY);
	if (fd<0)
		throw FileException("Cannot open dataset");
	MultithreadedDataset Temp;
	Temp.LoadHeader(fd,nRows,nCols);
	close(fd);
}

void MultithreadedDataset::GlobalFetchRow(long Position,DynamicArray<float> &row) {
	int nCols=GetColCount();
	ASSERT(Position>=0 && Position<GetRowCount());
	const float *src=GetRowNew(Position);
	for(int i=0;i<nCols;i++)
		row[i]=src[i];
}

void MultithreadedDataset::Save(char* fname) {
	FILE *fd=fopen(fname,"wb");
	int nCols=GetColCount();
	long nRows=GetRowCount();
	if (fd==NULL)
		throw FileException("Cannot open MultithreadedDataset");

	if( fwrite(&nTotalRows,sizeof(nTotalRows),1,fd)!=1
	   || fwrite(&nCols,sizeof(nCols),1,fd)!=1)
		  throw FileException("fread failed");
	   	
	for(long i=0;i<nRows;i++) {
		float *row = GetRowNew(i);
		if (fwrite(row,sizeof(float),nCols,fd)!=(unsigned)nCols)
			throw FileException("Cannot write row");			
//		TRACE2("Row %d Addr %x\n",i,pData);
	}
	fclose(fd);	
}

void MultithreadedDataset::LoadSelectedRows(const char *fname, DynamicArray<float> &FlatVector, const DynamicArray<long> &ObjNums) {




    MultithreadedDataset TempData;
    int nCols;
    long nTotalRows;


	FILE *fd=fopen(fname,"rb");
	if (fd==NULL)
		throw FileException("Cannot open dataset");

    TempData.LoadHeader(fileno(fd),nTotalRows,nCols);
    fseek(fd,0L,SEEK_SET);


    long nRequestedRows=ObjNums.GetSize();

	FlatVector.SetSize(nCols*nRequestedRows);

	for(long i=0;i<nRequestedRows;i++) {
		long Offset=i*nCols;
		long RequestedRow=ObjNums[i];
		fseek(fd,2*sizeof(int)+sizeof(float)*(long)RequestedRow*(long)nCols,SEEK_SET);
		fread(FlatVector.GetData()+Offset,sizeof(float),nCols,fd);
	}
	fclose(fd);
}

void ReadAssignment(std::string AssignmentName,NoInitArray<int> &Assignment) {
    FILE *pF= fopen(AssignmentName.c_str(),"rb");
    if (pF==nullptr)
        throw FileException("Cannot open cluster assignment file");
    int smallSize;
    long Size;
    fread(&smallSize,sizeof(smallSize),1,pF);
    if (smallSize==-1)
        fread(&Size,sizeof(Size),1,pF);
    else
        Size=smallSize;
    Assignment.SetSize(Size);
    fread(Assignment.GetData(),sizeof(int),Size,pF);
    for(long i=0;i<Size;i++)
        Assignment[i]-=1;

}