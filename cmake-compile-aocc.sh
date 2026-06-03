#!/bin/bash


export CXX=clang++ #AMD AOCC is run as clang
export CC=clang
DYNAMIC=OFF
MKL=OFF
AOCL=ON
for OPENMP in ON OFF
do
    rm -rf build
    mkdir build
    cd build
    clear
    cmake .. -DCMAKE_BUILD_TYPE=Release -DMKL_DIR=${MKLROOT}/lib/cmake/mkl -DUSE_MKL=${MKL} -DUSE_AOCL=${AOCL} -DUSE_OPENMP=${OPENMP} -DUSE_DYNAMIC=${DYNAMIC}
    cmake --build . -j `nproc`  -- VERBOSE=1
    cmake --install .
    cd ..
done
