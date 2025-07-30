#!/bin/bash


export CXX=icpc
export CC=icc
DYNAMIC=OFF
MKL=ON
OPENMP=ON
rm -rf build
mkdir build
cd build
clear
cmake .. -DCMAKE_BUILD_TYPE=Release -DMKL_DIR=${MKLROOT}/lib/cmake/mkl -DUSE_MKL=${MKL} -DUSE_OPENMP=${OPENMP} -DUSE_DYNAMIC=${DYNAMIC}
cmake --build . -j `nproc`  -- VERBOSE=1
cmake --install .
cd ..
