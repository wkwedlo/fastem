//
// Created by wkwedlo on 16.11.24.
//
#include <stdexcept>

#include "BlockManager.h"
#include "../Util/OpenMP.h"


void BlockManagerBase::ComputeEMBlockCounts(long nRows, int nCols, int K, double L3,int &nDensBlocks, int &nMStepBlocks) {
    L3*=(1024*1024);
    nDensBlocks=(double)nRows*(16*(double)nCols+16*(double)K)/L3*1.2;
    if (nDensBlocks<1)
        nDensBlocks=1;
    nMStepBlocks=nDensBlocks;
}

void BlockManagerBase::ParseEMBlockParams(const char *BlockParams, long nRows, int nCols, int K,int &nDensBlocks, int &nMStepBlocks) {
    double L3;
    if (BlockParams==nullptr) {
        nDensBlocks=nMStepBlocks=1;
        return;
    }
    if (sscanf(BlockParams,"automax:%lf",&L3)==1) {
        ComputeEMBlockCounts(nRows, nCols, K, L3, nDensBlocks, nMStepBlocks);
    } else if (sscanf(BlockParams,"%d:%d",&nDensBlocks,&nMStepBlocks)!=2)
        throw std::invalid_argument("Bad value of MStepLoopParam");
}


BlockManager::BlockManager(long anRows,long anBlocks) {
    nRows=anRows;
    nBlocks=anBlocks;
    BlockSize=nRows/nBlocks;
    BlockRemainder=nRows % nBlocks;

}

void BlockManager::GetBlock(long k, long &StartRow, long &Count) const {
    StartRow= k * BlockSize;
    Count=BlockSize;
    if (k == nBlocks - 1)
        Count+=BlockRemainder;
}

void BlockManager::GetThreadSubblock(long BlockStartRow, long BlockCount, long &StartRow, long &Count) const {
    FindOpenMPItems(BlockCount,StartRow,Count);
}

long BlockManager::GetThredMaxSubBlockSize() const {
    long MaxBlockSize=GetMaxBlockSize();
    long ThreadMaxBlockSize;
    long Offset;
    FindOpenMPItems(MaxBlockSize,Offset,ThreadMaxBlockSize);
    return ThreadMaxBlockSize;
}

SimpleBlockManager::SimpleBlockManager(long anRows, long anBlocks) {
    nBlocks=anBlocks;
    nRows=anRows;
}

void SimpleBlockManager::GetThreadPartition(long &StartRow, long &PartitionCount) const {
    FindOpenMPItems(nRows,StartRow,PartitionCount);
}

void SimpleBlockManager::GetThreadBlock(long k, long PartitionCount,long &Offset,long &Count) const {
    FindOpenMPItems(PartitionCount,Offset,Count,nBlocks,k);
}

long SimpleBlockManager::GetMaxThreadBlockSize() const {
    long StartRow,PartitionCount;
    FindOpenMPItems(nRows,StartRow,PartitionCount);
    long Offset,Count;
    FindOpenMPItems(PartitionCount,Offset,Count,nBlocks,0);
    return Count+1;
}



