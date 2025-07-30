/*
 * Compiler.cpp
 *
 *  Created on: Feb 6, 2018
 *      Author: wkwedlo
 */
#include "StdOut.h"
#include "MultithreadedDataset.h"
#include "OpenMP.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

void CompileHeader(char *Exec) {
	kma_printf("%s compilation time: %s %s\n",Exec,__DATE__,__TIME__);
#ifdef BRANCH
	kma_printf("Git branch: %s\n", BRANCH);
#endif

#if defined(__INTEL_COMPILER)
	kma_printf("Intel compiler\n");
#elif defined (__PGI)
	kma_printf("PGI compiler\n");
#else
	kma_printf("g++ compiler\n");
#endif

	kma_printf("Instruction set: ");
#if defined(__AVX2__)
	kma_printf("AVX2\n");
#elif defined(__AVX__)
	kma_printf("AVX\n");
#elif defined(__SSE4_2__)
	kma_printf("SSE4.2\n");
#else
	kma_printf("other\n");
#endif

#ifdef EIGEN_USE_MKL_ALL
	kma_printf("Using Intel MKL: yes\n");
#else
	kma_printf("Using Intel MKL: no\n");
#endif
	kma_printf("sizeof(OPTFLOAT): %d\n",(int)sizeof(OPTFLOAT));
	kma_printf("sizeof(EXPFLOAT): %d\n",(int)sizeof(EXPFLOAT));
	kma_printf("OpenMP loop scheduling policy: %s\n",TOSTRING(OMPDYNAMIC));

}

