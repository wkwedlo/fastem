// -*- C++ -*-


#ifndef __ARRAY_H

#include "Debug.h"
#include <cstring>
#include <cstdlib>
#include <vector>

#define __ARRAY_GROW_BY 16;
#define __ARRAY_H


//#define _DEBUG_FUNCTION_CALLS

#ifdef _DEBUG
#define _RANGE_CHECK_
#else
#undef  _DEBUG_FUNCTION_CALLS
#endif



void CheckArrayRange(long i,long Size);
void NumaScan(char *ptr,long nBytes);


// Supress inline expansion if the range check is enabled

#ifdef _RANGE_CHECK_
#define ARR_INLINE
#else
#define ARR_INLINE  inline
#endif






/** Template for a C-style array, with dynamic memory management
    it is based on std::vector, but its index operator checks for out-of-bounds access
    in debug mode and does not check in release mode
*/
template <typename T, typename Allocator=std::allocator<T> > class DynamicArray
{

protected:

	std::vector<T,Allocator> vec;
public:
    /// Constructs array with initial size = 0
    DynamicArray() {}
    explicit DynamicArray(long aSize);
    explicit DynamicArray(const DynamicArray<T,Allocator> &A);


    DynamicArray &operator=(const DynamicArray<T,Allocator> &arr);

    /// Destructor frees all the memory
    ~DynamicArray() {}
    /// Changes the size of the array, preserving data if possible
    long GetSize() const {return vec.size();}
    void SetSize(long aSize,long GrowBy=-1);

    T *GetData() {return vec.data();}
    const T* GetData() const {return vec.data();}

    /// array index operator
    T & operator[](long Idx);
    /// array index operator for read access
    const T & operator[](long Idx) const;


    /// Appends Src to the end of array
    long Append(DynamicArray<T,Allocator> &Src);
	/// Appends Elem to the end of array
    void Add(const T &Elem);
	/// Removes Count elements starting from position Idx    
    void RemoveAt(long Idx,long Count=1);
  
    void InsertAt(long Idx, T &Elem,long Count=1);

};



template <typename T,typename Allocator> DynamicArray<T,Allocator>::DynamicArray(long aSize)
: vec(aSize)
{
	   
}


template <typename T,typename Allocator> DynamicArray<T,Allocator>::DynamicArray(const DynamicArray<T,Allocator> & A) : vec(A.vec)
{
}




template <typename T,typename Allocator> DynamicArray<T,Allocator> & DynamicArray<T,Allocator>::operator=(const DynamicArray<T,Allocator> &A)
{
	vec=A.vec;
    return *this;
}


template <typename T,typename Allocator> ARR_INLINE T& DynamicArray<T,Allocator>::operator[](long Idx)
{
#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,vec.size());
#endif
    return vec[Idx];
}

template <typename T,typename Allocator> ARR_INLINE const T& DynamicArray<T,Allocator>::operator[](long Idx) const
{
#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,vec.size());
#endif
    return vec[Idx];
}

template <typename T,typename Allocator> long DynamicArray<T,Allocator>::Append(DynamicArray<T,Allocator> &Src)
{

	long Ret=vec.size();
	vec.resize(Ret+Src.GetSize());
    for(long i=0;i<Src.GetSize();i++)
		vec[Ret+i]=Src[i];
    return Ret;
}

template <typename T,typename Allocator> void DynamicArray<T,Allocator>::SetSize(long aSize,long GrowBy)
{
	vec.resize(aSize);
}


template <typename T,typename Allocator> ARR_INLINE void DynamicArray<T,Allocator>::Add(const T &Elem)
{
	vec.push_back(Elem);
}


template <typename T,typename Allocator> void DynamicArray<T,Allocator>::RemoveAt(long Idx,long Count)
{
#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,vec.size());
#endif
    vec.erase(vec.begin()+Idx,vec.begin()+Idx+Count);
}



template <typename T,typename Allocator> void DynamicArray<T,Allocator>::InsertAt(long Idx, T &Elem,long Count)
{

#ifdef _RANGE_CHECK_
    CheckArrayRange(Idx,vec.size());
#endif
    vec.insert(vec.begin()+Idx,Count,Elem);
}

template <typename T> class CacheAlignedAllocator  {
    public:
    using value_type = T;
    T* allocate(size_t n, const void* hlong = 0) {
        constexpr size_t alignment=64;
        size_t nbytes=sizeof(T)*n;
        size_t remainder=nbytes % alignment;
        if (remainder>0)
            nbytes+=(alignment-remainder);
        ASSERT(nbytes % alignment==0);
        return (T *)std::aligned_alloc(alignment,nbytes);
    }
    void deallocate(T* p, size_t n) {
        free(p);
    }

};


template <typename T,typename Allocator=CacheAlignedAllocator<T>> class CacheAlignedArray : public DynamicArray<T,Allocator > {
public:
    explicit CacheAlignedArray (long aSize) : DynamicArray<T, Allocator>(aSize) {}
    explicit CacheAlignedArray(const CacheAlignedArray<T> &A) : DynamicArray<T,Allocator >(A) {}
};





template <typename T>
class NoInitAllocator {
public:
    using value_type = T;

    T* allocate(size_t n, const void* hlong = 0) {
        return (T*)std::malloc(n*sizeof (T));
    }

    void deallocate(T* p, size_t n) {
        free(p);
    }

        template <class U, class... Args>
    void construct(U* p, Args&&... args) noexcept {}

    template <class U>
    void destroy(U* p) noexcept { }
};

/**
 * @brief array that does not initialize its elements, unlike DynamicArray and std::vector
 */
template <typename T> class NoInitArray : public DynamicArray<T,NoInitAllocator<T> > {
    public:
    using DynamicArray<T,NoInitAllocator<T>>::DynamicArray;
};
#endif // __ARRAY_H
