#include <mm_malloc.h>

#include "CovMatrixArray.h"



void CovMatrixArray::SegmentedReduction(DynamicArray<CovMatrixArray> &Data) {

    long nSize = Data[0].GetPackedSize();
    OPTFLOAT* __restrict__ acc = Data[0].GetData();

    constexpr int BLOCK_SIZE = 2048;

#pragma omp parallel default(none) firstprivate(nSize,acc) shared(Data)
    {
        int tid = omp_get_thread_num();
        int nThreads = omp_get_num_threads();

        long ChunkSize = (nSize + nThreads - 1) / nThreads;
        long StartIdx = tid * ChunkSize;
        long EndIdx = std::min(StartIdx + ChunkSize, nSize);

        for (long b = StartIdx; b < EndIdx; b += BLOCK_SIZE) {
            long BlockEnd = std::min(b + BLOCK_SIZE, EndIdx);

            for (int t = 1; t < nThreads; t++) {
                const OPTFLOAT * __restrict__ src = Data[t].GetData();

#pragma omp simd
                for (long i = b; i < BlockEnd; i++) {
                    acc[i] += src[i];
                }
            }
        }
    }
}

CovMatrixArray::CovMatrixArray(int aK, int anCols) : nCols(anCols), K(aK) {
    SetSize(aK, anCols);
}

void CovMatrixArray::SetSize(int aK, int anCols) {
    ASSERT(m_data== nullptr);
    ASSERT(maps.empty());
    nCols=anCols;
    K=aK;
    long RowBytes=nCols*sizeof(OPTFLOAT);
    long AlignedRowBytes=(RowBytes + kAlignBytes - 1) & ~(kAlignBytes - 1);
    nStride=AlignedRowBytes/sizeof(OPTFLOAT);
    size_t TotalBytes= (size_t)K * (size_t)nStride * (size_t)nCols*(size_t)sizeof(OPTFLOAT);
    m_data=(double *)_mm_malloc(TotalBytes,kAlignBytes);
    maps.reserve(K);
    for(int j=0;j<K;j++) {
        size_t offset=(size_t)j*(size_t)nStride*(size_t)nCols;
        maps.emplace_back(m_data + offset,nCols,nCols,Eigen::OuterStride<>(nStride));
    }
#ifdef _DEBUG
    for(int j=0;j<K;j++) {
        ASSERT(((unsigned long)(maps[j].data()) % kAlignBytes) == 0);
    }
#endif
}

    CovMatrixArray::~CovMatrixArray() {
    if (m_data!= nullptr)
        _mm_free(m_data);
}

CovMatrixArray::CovMatrixArray(CovMatrixArray &&other) noexcept : m_data(other.m_data), nCols(other.nCols),
    nStride(other.nStride), K(other.K), maps(std::move(other.maps)) {
    other.m_data = nullptr;
    other.nCols = 0;
    other.nStride = 0;
    other.K = 0;
}


CovMatrixArray & CovMatrixArray::operator+=(const CovMatrixArray &other) {
    ASSERT(GetPackedSize()==other.GetPackedSize());

    size_t PackedSize=GetPackedSize();
    for(size_t i=0;i<PackedSize;i++) {
        m_data[i]+=other.m_data[i];
    }
    return *this;
}

void CovMatrixArray::ClearData() {
    size_t TotalSize = (size_t)K * (size_t)nStride * (size_t)nCols;
    for (size_t i = 0; i < TotalSize; i++) {
        m_data[i] = 0.0;
    }
}

void CovMatrixArray::PackInPlace() {
    OPTFLOAT * pWriter = m_data;
    for (int j = 0; j < K; j++) {
        OPTFLOAT *pMatrix = m_data + (j * nCols * nStride);

        for (int r = 0; r < nCols; ++r) {
            OPTFLOAT * pRow = pMatrix + (r * nStride);
            for (int c = r; c <nCols; ++c) {
                *pWriter++ = pRow[c];
            }
        }
    }
}


void CovMatrixArray::UnpackInPlace() {
    size_t PackedSize = (size_t)nCols * ((size_t)nCols + 1) / 2;
    for (int i = K - 1; i >= 0; i--) {
        const OPTFLOAT* pSrc = m_data + ((i + 1) * PackedSize);
        OPTFLOAT* pMatrixBase = m_data + (i * nCols * nStride);
        for (long r = nCols - 1; r >= 0; r--) {
            OPTFLOAT * pRow = pMatrixBase + (r * nStride);
            for (long c = nCols - 1; c >= r; c--) {
                pRow[c] = *(--pSrc);
            }
        }
    }
}

