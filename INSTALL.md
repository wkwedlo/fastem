## Recommended operating system
- Should work on any Linux distribution, the code was initially developed on Ubuntu 22.04
- Production runs for experiments reported in the paper were done on a cluster using Rocky Linux 8 in compute nodes.


## Requirements 
- libnuma-dev
- Cmake >= 3.25. See instruction [here](https://apt.kitware.com/) for the latest version for Ubuntu 22.04 
- Intel OneAPI toolkit. See instruction [here](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-2/apt.html). Both the basekit and the HPC kit are required. Please install 
  an older version using the command `apt install intel-basekit-2023.2.0 intel-hpckit-2023.2.0` (for Debian-based systems) 
as the newest versions lack `icpc` compiler and will not compile the code.
- Alternatively you should be able to use GCC if you comment out the CXX and CC variable in the shell script. In this case
you still need Intel OneAPI for the MKL and probably for the MPI library.
- In theory, it should be possible to build with another BLAS library but this would require some changes in the code and heavy changes in the CMake files.

## Installation of git submodule with Eigen C++ library
```
git submodule update --init
```


## Compilation and install
```
#Initialize oneAPI e.g.,    . /opt/intel/oneapi/setvars.sh
./cmake-compile-local-default.sh 
```
The compiled and installed binaries are in the `~/bin` directory. The following binaries should be created (all have long suffixed names depending on project options):
- eminit_*  creates the initial condition for the EM algorithm and stores it in a file with the name specified in the command line (OpenMP only variant or sequential variant depending on the USE_OPENMP flag)
- em2_*  runs the EM algorithm (OpenMP only variant or sequential variant depending on the USE_OPENMP flag)
- mpiem2_*  runs the EM algorithm (MPI/OpenMP variant or MPI only variant depending on the USE_OPENMP flag)


## Environment variables
- ICCOPT: CPU architecture optimization options for the Intel compiler. For experiments reported in the paper (Intel Xeon Platinum 8268) we set -march=cascadelake
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
- -c 200: number of micxture components
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
