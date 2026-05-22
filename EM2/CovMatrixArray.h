
//
// Created by wkwedlo on 10.12.2025.
//

#ifndef CLUST_COVMATRIXARRAY_H
#define CLUST_COVMATRIXARRAY_H

#include <vector>

#include "../Util/EigenUtil.h"


class CovMatrixArray {

    OPTFLOAT *m_data= nullptr;
    int nCols=0;
    int  nStride=0;
    int  K=0;
    std::vector<EigenAlignedMap> maps;
public:
    static void SegmentedReduction(DynamicArray<CovMatrixArray> &Data);

    CovMatrixArray(int aK, int anCols);
    void SetSize(int aK, int anCols);
    CovMatrixArray() {}
    ~CovMatrixArray();
    CovMatrixArray(CovMatrixArray &&other) noexcept;

    CovMatrixArray(const CovMatrixArray &other)=delete;
    CovMatrixArray& operator=(const CovMatrixArray&)=delete;


    const EigenAlignedMap & operator[](int index) const {
        return maps[index];
    }
    EigenAlignedMap & operator[](int index) {
        return maps[index];
    }
    int GetComponentCount() const {return K;}
    int GetDimension() const {return nCols;}
    int GetStride() const {return nStride;}
    int GetPackedSize() const {
        return K * (nCols+1) * nCols/2;
    }
    OPTFLOAT *GetData() const {return m_data;}
    CovMatrixArray &operator+=(const CovMatrixArray &other);
    void ClearData();
    void PackInPlace();
    void UnpackInPlace();
};


#endif //CLUST_COVMATRIXARRAY_H