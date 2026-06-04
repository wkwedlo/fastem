## Recommended operating system
- Should work on any Linux distribution, the code was initially developed on Ubuntu 22.04 LTS
- Experiments for Intel Xeon architecture reported in the paper were run on a cluster using Rocky Linux 8 in Intel compute nodes. The compiler was `icpc version 2021.10.0 (gcc version 8.5.0 compatibility)`
- Experiments for AMD EPYC architecture reported in the paper were run on a system with Ubuntu 22.04 LTS. The compiler was 
AMD AOCC (`AMD clang version 17.0.6 (CLANG: AOCC_5.0.0-Build#1377 2024_09_24)`). The BLAS routines were provided by AMD AOCL library compiled by the AOCC compiler. The MPI library for AMD EPYC was Intel MPI from the oneAPI package.



## Requirements 
- libnuma-dev
- Cmake >= 3.26. See instruction [here](https://apt.kitware.com/) for the latest version for Ubuntu 22.04 
- Intel OneAPI toolkit for Intel systems. See instruction [here](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-2/apt.html). Both the basekit and the HPC kit are required. To replicate exactly the experiments from the paper using icpc compiler please install an older version using the command `apt install intel-basekit-2023.2.0 intel-hpckit-2023.2.0` (for Debian-based systems) 
as the newest versions lack `icpc`. Note: icpc compiler breaks down compilation on 
newer Linux systems. In such case use Intel icpx or g++ compiler.
- [AMD AOCC compiller](https://www.amd.com/en/developer/aocc.html) and [AMD AOCL library](https://www.amd.com/en/developer/aocl.html) for AMD EPYC systems. Note that you still need an MPI library (we used MPI from Intel oneAPI in our EPYC experiments)
## Installation of git submodule with Eigen C++ library
```
git submodule update --init
```


## Compilation and install
### Intel icpc compiler
Use the environment variable `ICCOPT` to set CPU architecture optimization options. For experiments reported in the paper (Intel Xeon Platinum 8268) we set `-march=cascadelake`

```
#Initialize oneAPI e.g.,    . /opt/intel/oneapi/setvars.sh
./cmake-compile-icpc.sh 
```

### GCC compiler
Use the environment variable `GCCOPT` to set CPU architecture optimization options. For experiments reported in the paper (Intel Xeon Platinum 8268) we set `-march=cascadelake` for Intel Xeon nodes.

```
#Initialize oneAPI e.g.,    . /opt/intel/oneapi/setvars.sh
./cmake-compile-gcc.sh 
```

### Intel ICPX compiler
Use the environment variable `ICXOPT` to set CPU architecture optimization options. For experiments reported in the paper (Intel Xeon Platinum 8268) we set `-march=cascadelake` for Intel Xeon nodes.

```
#Initialize oneAPI e.g.,    . /opt/intel/oneapi/setvars.sh
./cmake-compile-icpx.sh 
```

### AMD AOCC compiler with AMD AOCL library 
This compiler is Clang++ variant heavily modified by AMD. You probably have to set the `LD_LIBRARY_PATH` variable to add runtime libraries required by the compiler.

Use the environment variable `AOCCOPT` to set CPU architecture optimization options. For experiments reported in the paper AMD EPYC 7402) we set `-march=znver2`. You have to set the `AOCL_ROOT` environment variable to the location of AOCL. For the newest AOCL 5.3 you **have to use the single-threaded (ST) variant**



```
#Initialize oneAPI (required for MPI) or setup your own MPI library    
 . /opt/intel/oneapi/setvars.sh 
./cmake-compile-aocc.sh 
```





## Location and the names of compiled executables 
The compiled and installed binaries are in the `~/bin` directory. The following binaries should be created (all have long suffixed names depending on project options, e.g., the compiler, OpenMP setting):
- eminit_*  creates the initial condition for the EM algorithm and stores it in a file with the name specified in the command line (OpenMP only variant or sequential variant depending on the USE_OPENMP flag)
- em2_*  runs the EM algorithm (OpenMP only variant or sequential variant depending on the USE_OPENMP flag)
- mpiem2_*  runs the EM algorithm (MPI/OpenMP variant or MPI only variant depending on the USE_OPENMP flag)


## Environment variables
- GCCOPT: CPU architecture optimization options for the GNU compiler. 

## Data format
The data matrix X is stored in a binary file in the following format:
```
Header
row0
row1
row2
...
rowN 
```
where each row is a vector of d floats (single precision). Before the EM run these vectors are **up-converted to double precision**.
The header consists of either two 32-bit integers (N,d) or if N==-1 is followed by a 64-bit integer (N) for datasets where N is larger than 2^31-1.

## Example program options to replicate one of the experiments in the paper


### The eminit program run to generate the initial solution for the EM algorithm using K-means clustering initialized with the Kmeans|| method

We assume that the dataset X is stored in a file called emnistpca300.bin (first 300 PCA components of the EMNIST dataset) 
The output file with the initial solution is called emnistpca300_k200_kmeansoror.cmp. 

```
export OMP_NUM_THREADS=48
export OMP_CPU_BIND=close
eminit_icc_openmp_release_double_mkl_master -c 200 emnistpca300.bin -I kmeansoror -o emnistpca300_k200_kmeansoror.cmp -r 3
```
Command line options:
- -c 200: number of mixture components
- -I kmeansoror: initialization method
- -o emnistpca300_k200_kmeansoror.cmp: output file with the initial solution
- -r 3: the seed of the RNG.


### The FastEM algorithm run for the emnist dataset (stored in a file emnistpca300.bin) and K=200

We assume that a SLURM script will place 4 MPI processes on each node and each process will use 12 threads on a 48 core node with 4 NUMA domains.

```
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export I_MPI_PIN_DOMAIN=omp

mpirun mpiem2_icc_openmp_release_double_mkl_master -c 200 -v 0 emnistpca300.bin -I emnistpca300_k200_kmeansoror.cmp -A simplematrixem -N automax:18 -f 1e-11 -m 100 -S -R log2 -b 5 -g 0.001
```

Command line options:
- -c 200: number of mixture components
- -v 0: verbosity level
- -I emnistpca300_k200_kmeansoror.cmp: input file with the initial solution
- -A simplematrixem: algorithm to use. This chooses the FastEM algorithm from the paper.
- -N automax:18: the amount (in MB) of the L3 cache memory available to each MPI process. The number 18 is equal to approximately total L3 cache (71.5 MB)  in a cluster node divided by the number of MPI processes (4) on the node.
- -f 1e-11: the convergence criterion
- -m 100: the maximum number of iterations
- -S: disable SVD decomposition of the covariance matrices to detect singularity (not needed as the regularization is used)
- -R log2: the reduction method. The log2 method is the binary tree reduction is used in the paper.
- -b 5: the number of 'burn-in' iterations performed before starting execution time measurement. The burn-in iterations are not included in the final result.
- -g 0.001: the regularization parameter. The value of 0.001 is used in the paper.
