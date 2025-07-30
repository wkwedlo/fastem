//
// Created by wkwedlo on 16.11.24.
//

#ifndef CLUST_BLOCKMANAGER_H
#define CLUST_BLOCKMANAGER_H

#include "../Util/Array.h"


class BlockManagerBase
{
    protected:
        static void ComputeEMBlockCounts(long nRows,int nCols,int K,double L3,int &nDensBlocks,int &nMStepBlocks);
    public:
        static void ParseEMBlockParams(const char *BlockParams,long nRows,int nCols,int K,int &nDensBlocks,int &nMStepBlocks);
};


class BlockManager : public BlockManagerBase {

protected:
    long nBlocks;
    long nRows;
    long BlockSize;
    long BlockRemainder;

public:
    BlockManager(long anRows,long anBlocks);
    void GetBlock(long k, long &StartRow, long &Count) const;
    void GetThreadSubblock (long BlockStartRow, long BlockCount, long &StartRow, long &Count) const;
    long GetThredMaxSubBlockSize() const;
    long GetMaxBlockSize() const {return BlockSize+BlockRemainder;}
    long GetBlockCount() const {return nBlocks;}
    static void ParseEMBlockParams(const char *BlockParams,long nRows,int nCols,int K,int &nDensBlocks,int &nMStepBlocks);
};

class SimpleBlockManager : public BlockManagerBase {
    long nBlocks;
    long nRows;

    long BlockSize;
    long BlockRemainder;


public:
    SimpleBlockManager(long anRows,long anBlocks);
    long GetBlockCount() const {return nBlocks;}
    void GetThreadPartition(long &StartRow, long &PartitionCount) const;
    void GetThreadBlock(long k, long PartitionCount,long &Offset,long &Count) const;
    long GetMaxThreadBlockSize() const;
};
#endif //CLUST_BLOCKMANAGER_H
