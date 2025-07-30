/*
 * DistributedMultithreadedDataset.cpp
 *
 *  Created on: Oct 8, 2015
 *      Author: wkwedlo
 */
#include <mpi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "DistributedMultithreadedDataset.h"
#include "../Util/FileException.h"
#include "../MPIKmeans/MPIUtils.h"

DistributedMultithreadedDataset::DistributedMultithreadedDataset(int anBlocks) : MultithreadedDataset(anBlocks) {
	int Size,Rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);

	Offsets.SetSize(Size);
	Counts.SetSize(Size);
}


void DistributedMultithreadedDataset::ComputeDataPositions(int Size) {

	for(int Rank=0;Rank<Size;Rank++) {
		long SubSize=nTotalRows/Size;
		long SubRemainder=nTotalRows % Size;
		long Offset;
		if (Rank<SubRemainder) {
			SubSize++;
			Offset=SubSize*Rank;
		} else {
			Offset=SubSize*Rank+SubRemainder;
		}
		Counts[Rank]=SubSize;
		Offsets[Rank]=Offset;
	}
	//TRACE3("MPI Process %d row offset :%d row count %d\n",Rank,RowOffset,RowCount);
}
void DistributedMultithreadedDataset::QueryHeader(char *fname,long &nRows,int &nCols) {
	int fd=open(fname,O_RDONLY);
	if (fd<0)
		throw FileException("Cannot open dataset");
	DistributedMultithreadedDataset Temp;
	Temp.LoadHeader(fd,nRows,nCols);
	Temp.nTotalRows=nRows;
	int Size,Rank;
	MPI_Comm_size(MPI_COMM_WORLD,&Size);
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	Temp.ComputeDataPositions(Size);
	close(fd);
	nRows=Temp.Counts[Rank];
}

void DistributedMultithreadedDataset::GlobalFetchRow(long Position,DynamicArray<float> &row) {
	ASSERT(Position>=0 && Position<nTotalRows);
	int Rank,Size,Root=-1;
	int nCols=GetColCount();
	MPI_Comm_size(MPI_COMM_WORLD,&Size);
	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);

	for(int i=0;i<Size;i++) {
		if (Offsets[i]<=Position && Position<Offsets[i]+Counts[i])
			Root=i;
	}
	if (Rank==Root) {
		long MyRow=Position-Offsets[Rank];
		ASSERT(MyRow>=0);
		ASSERT(MyRow<=GetRowCount());
		for(int i=0;i<nCols;i++)
			row[i]=(*this)(MyRow,i);
	}
	MPI_Bcast(row.GetData(),nCols,MPI_FLOAT,Root,MPI_COMM_WORLD);
}


void DistributedMultithreadedDataset::Load(char *fname) {
	int Size,Rank;

	MPI_Comm_rank(MPI_COMM_WORLD,&Rank);
	MPI_Comm_size(MPI_COMM_WORLD,&Size);

	int fd=open(fname,O_RDONLY);
	if (fd<0)
		throw FileException("Cannot open dataset");

	long nRows;
    int nCols;
	LoadHeader(fd,nRows,nCols);
	nTotalRows=nRows;
	close(fd);
	ComputeDataPositions(Size);
	PartialLoad(fname,Offsets[Rank],Counts[Rank],nCols);
	ComputeStatistics();
}

void DistributedMultithreadedDataset::ReduceSumStatistics(DynamicArray<OPTFLOAT> &Vec) {
	DynamicArray<OPTFLOAT> SendVec(Vec);
	int nCols=GetColCount();
	MPI_Allreduce(SendVec.GetData(),Vec.GetData(),nCols,OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
}

void DistributedMultithreadedDataset::ReduceStatisticsStep1(DynamicArray<OPTFLOAT> &MeanSum, DynamicArray<float> &Min,
	DynamicArray<float> &Max) {
	DynamicArray<OPTFLOAT> SendVec(MeanSum);
	int nCols=GetColCount();
	MPI_Allreduce(SendVec.GetData(),MeanSum.GetData(),nCols,OptFloatType(),MPI_SUM,MPI_COMM_WORLD);
	DynamicArray<float> SendVecflt(Min);
	MPI_Allreduce(SendVecflt.GetData(),Min.GetData(),nCols,MPI_FLOAT,MPI_MIN,MPI_COMM_WORLD);
	SendVecflt=Max;
	MPI_Allreduce(SendVecflt.GetData(),Max.GetData(),nCols,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
}

void DistributedMultithreadedDataset::ReduceStatisticsStep2(DynamicArray<OPTFLOAT> &StdSum) {
	DynamicArray<OPTFLOAT> SendVec(StdSum);
	int nCols=GetColCount();
	MPI_Allreduce(SendVec.GetData(),StdSum.GetData(),nCols,OptFloatType(),MPI_SUM,MPI_COMM_WORLD);

}
