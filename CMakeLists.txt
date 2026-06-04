cmake_minimum_required(VERSION 3.26)
project(clust LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_PREFIX $ENV{HOME})


function(add_likwid TARGET)
    if (USE_LIKWID)
        target_link_directories(${TARGET} PUBLIC $ENV{LIKWID_HOME}/lib)
        target_include_directories(${TARGET} PUBLIC $ENV{LIKWID_HOME}/include)
        target_link_libraries(${TARGET} PUBLIC likwid)
        target_compile_definitions(${TARGET} PUBLIC LIKWID)
    endif()
endfunction()


function(add_itac TARGET)
    if (USE_ITAC)
        target_link_directories(${TARGET} PUBLIC $ENV{VT_ROOT}/lib)
        target_include_directories(${TARGET} PUBLIC $ENV{VT_ROOT}/include)
        target_link_libraries(${TARGET} PUBLIC VT dwarf elf vtunwind pthread ${ZLIB_LIBRARIES})
        target_compile_definitions(${TARGET} PUBLIC INTEL_ITAC)
    endif()
endfunction()

function(add_eigen_blas TARGET)
    target_include_directories(${TARGET} PUBLIC ${EIGEN_PATH})
    target_compile_definitions(${TARGET} PUBLIC ${EIGEN_OPTIONS})
    if (USE_MKL)
        target_include_directories(${TARGET} PUBLIC ${MKL_INCLUDE})
        target_link_libraries(${TARGET} PUBLIC MKL::MKL)
    elseif (USE_AOCL)
        target_include_directories(${TARGET} PUBLIC $ENV{AOCL_ROOT}/include)
        target_link_directories(${TARGET} PUBLIC $ENV{AOCL_ROOT}/lib)
        target_link_libraries(${TARGET} PUBLIC amdlibm blis flame)
    else()
        message(FATAL_ERROR "No BLAS library specified")
    endif()
endfunction()

find_package(Git REQUIRED)
execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_definitions(BRANCH=\"${GIT_BRANCH}\")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-flto -Ofast -fno-finite-math-only $ENV{GCCOPT} -fopenmp-simd -g3")
set(EIGEN_PATH ./external/eigen)

option(USE_OPENMP "Switches OpenMP compilation" ON)
option(USE_DOUBLEFLOAT "Uses double precision floating point numbers" ON)
option(USE_MKL "Uses Intel MKL Library" ON)
option(USE_AOCL "Uses AMD AOCL Library" OFF)
option(USE_DYNAMIC "Uses schedule(guided) OpenMP loops" OFF)
option(USE_ITAC "Uses Intel Trace Collector library" OFF)
option(USE_ITT "Uses Intel Instrumentation and Tracing library" OFF)
option(USE_LIKWID "Uses Likwid performance monitoring tool" OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS_RELEASE "-qopenmp-simd -Ofast -fno-finite-math-only -ipo $ENV{ICXOPT} -g3")
    set(COMPILER_FIX _icx)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS_RELEASE "-qopenmp-simd -Ofast -fno-finite-math-only -ipo $ENV{ICCOPT} -g3")
    add_compile_options(-diag-disable=10441)
    add_link_options(-diag-disable=10441)
    set(COMPILER_FIX _icc)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND USE_AOCL) # Must be AMD AOCC
    set(CMAKE_CXX_FLAGS_RELEASE "-flto -Ofast -fno-finite-math-only $ENV{AOCCOPT} -fopenmp-simd -g3")
    set(COMPILER_FIX _aocc)
endif()

set(KMEANS_INIT_SOURCES Clust/CentroidVector.cpp Clust/CentroidVector.h
    Clust/KMeansInitializer.cpp Clust/KMeansInitializer.h
    Clust/KMeansOrOrInitializer.cpp Clust/KMeansOrOrInitializer.h
    Clust/PlusPlusInitializer/PlusPlusInitializer.h Clust/PlusPlusInitializer/PlusPlusInitializer.cpp)


set(UTIL_SOURCES Util/Array.cpp Util/Array.h Util/Compiler.cpp Util/Compiler.h Util/MultithreadedDataset.cpp
    Util/MultithreadedDataset.h Util/Debug.cpp Util/Debug.h Util/FileException.h Util/LargeMatrix.h Util/LargeVector.h
    Util/PrecisionTimer.cpp Util/PrecisionTimer.h Util/StdOut.cpp Util/StdOut.h Util/NumaAlloc.cpp Util/NumaAlloc.h
    Util/Rand.cpp Util/Rand.h Util/OMPReducer.h Util/ThreadAffinityInfo.cpp Util/ThreadAffinityInfo.h
    Util/Profiler.cpp Util/Profiler.h EM2/BlockManager.cpp EM2/BlockManager.h)


set(EMINIT_SOURCES ${KMEANS_INIT_SOURCES} ${UTIL_SOURCES} EM2/EMInitializer.cpp EM2/EMInitializer.h EM2/NormalDensity.cpp
        EM2/NormalDensity.h EM2/GaussianMixture.cpp  EM2/GaussianMixture.h Util/EigenUtil.cpp Util/EigenUtil.h
        EM2/EMException.cpp EM2/EMException.h
        Clust/KMAlgorithm.cpp Clust/KMAlgorithm.h Clust/CentroidRepair.cpp Clust/CentroidRepair.h
        Clust/KMeansWithoutMSE/KMeansWithoutMSE.cpp Clust/KMeansWithoutMSE/KMeansWithoutMSE.h
        Clust/KMeansReportWriter.cpp Clust/KMeansReportWriter.h
        Clust/OpenMPKMAReducer.cpp Clust/OpenMPKMAReducer.h)


set(EM_SOURCES ${EMINIT_SOURCES} EM2/EMAlgorithm.cpp EM2/EMAlgorithm.h
     EM2/MatrixEM.cpp EM2/MatrixEM.h Util/ITAC.h Util/Likwid.h EM2/CovMatrixArray.cpp EM2/CovMatrixArray.h
    EM2/OpenMPEMReducer.h EM2/OpenMPEMReducer.cpp
)


set(MPIEM_SOURCES ${EM_SOURCES}  MPIEM2/MPIJainEMInitializer.cpp MPIEM2/MPIJainEMInitializer.h
    MPIEM2/MPIMatrixEM.h MPIEM2/MPIMatrixEM.cpp MPIEM2/MPIUniversalReducer.h
    Util/MPIThreadAffinityInfo.cpp Util/MPIThreadAffinityInfo.h MPIKmeans/MPIRank0StdOut.cpp MPIKmeans/MPIRank0StdOut.h
    MPIKmeans/DistributedMultithreadedDataset.cpp MPIKmeans/DistributedMultithreadedDataset.h)


string(TOLOWER ${CMAKE_BUILD_TYPE} BUILDTYPE_FIX)
if (${CMAKE_BUILD_TYPE} STREQUAL Debug)
    set(EIGEN_OPTIONS -DEIGEN_DONT_PARALLELIZE -DEIGEN_RUNTIME_NO_MALLOC)
    add_compile_definitions(_DEBUG)
else()
    set(EIGEN_OPTIONS -DEIGEN_DONT_PARALLELIZE -DNDEBUG)
endif()

if (USE_DOUBLEFLOAT)
    set(FLOAT_FIX double)
    add_compile_definitions(DOUBLEFLOAT)
else()
    set(FLOAT_FIX float)
endif()

if (USE_DYNAMIC)
    add_compile_definitions(OMPDYNAMIC=schedule\(guided\))
    set(DYNAMIC_FIX _dynamic)
endif()

if (USE_ITT)
    add_compile_definitions(INTEL_ITT)
    find_library(ITT_LIB NAMES ittnotify PATHS $ENV{VTUNE_PROFILER_DIR}/lib64)
    link_libraries(${ITT_LIB})
    link_libraries(-ldl)
    include_directories($ENV{VTUNE_PROFILER_DIR}/include)
    message("Intel ITT" ${ITT_LIB})
    set(ITT_FIX _itt)
endif()

if (USE_LIKWID)
    set(ITT_FIX _likwid)
endif()

if (USE_MKL)
    set(MKL_THREADING sequential)
    set(MKL_INTERFACE_FULL intel_lp64)
    find_package(MKL REQUIRED)
    set(EIGEN_OPTIONS ${EIGEN_OPTIONS} -DEIGEN_USE_MKL_ALL)
    set(MKL_FIX _mkl)
endif()

if (USE_AOCL)
    set(EIGEN_OPTIONS ${EIGEN_OPTIONS} -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
    set(MKL_FIX _aocl)
endif()


if (USE_OPENMP)
    find_package(OpenMP)
#    add_compile_options(${OpenMP_CXX_FLAGS})
#    add_link_options(${OpenMP_CXX_FLAGS})
    link_libraries(OpenMP::OpenMP_CXX)
    set(OPENMP_FIX _openmp)
endif()

link_libraries(numa)

find_package(MPI REQUIRED)
if (MPI_FOUND)
    if (USE_ITAC)
        find_package(ZLIB REQUIRED)
        set(ITAC_FIX _itac)
    endif()
endif()

set(EXECUTABLE_SUFFIX ${ITT_FIX}${ITAC_FIX}${COMPILER_FIX}${OPENMP_FIX}${DYNAMIC_FIX}_${BUILDTYPE_FIX}_${FLOAT_FIX}${MKL_FIX}_${GIT_BRANCH})

set(EM2_EXECUTABLE em2${EXECUTABLE_SUFFIX})
set(EMINIT_EXECUTABLE eminit${EXECUTABLE_SUFFIX})
set(MPIEM2_EXECUTABLE mpiem2${EXECUTABLE_SUFFIX})

add_executable(${EM2_EXECUTABLE} ${EM_SOURCES} EM2/main.cpp EM2/EMArgs.cpp EM2/EMArgs.h Util/Likwid.h)
add_eigen_blas(${EM2_EXECUTABLE})
add_likwid(${EM2_EXECUTABLE})

add_executable(${EMINIT_EXECUTABLE} ${EMINIT_SOURCES} EM2/eminit.cpp)
add_eigen_blas(${EMINIT_EXECUTABLE})

install(TARGETS ${EM2_EXECUTABLE} ${EMINIT_EXECUTABLE})


if (MPI_FOUND)
    add_executable(${MPIEM2_EXECUTABLE} ${MPIEM_SOURCES} MPIEM2/main.cpp EM2/EMArgs.cpp EM2/EMArgs.h Util/Likwid.h)
    target_link_libraries(${MPIEM2_EXECUTABLE} PUBLIC MPI::MPI_CXX)
    target_include_directories(${MPIEM2_EXECUTABLE} PUBLIC  ${MPI_CXX_INCLUDE_DIRS})
    target_compile_definitions(${MPIEM2_EXECUTABLE} PUBLIC -D_MPI)
    add_eigen_blas(${MPIEM2_EXECUTABLE})
    add_itac(${MPIEM2_EXECUTABLE})
    add_likwid(${MPIEM2_EXECUTABLE})

    install(TARGETS  ${MPIEM2_EXECUTABLE})

endif()


