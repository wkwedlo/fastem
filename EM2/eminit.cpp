#include <stdio.h>
#include <unistd.h>

#include "../Util/OpenMP.h"
#include "../Util/Rand.h"
#include "../Util/StdOut.h"
#include "../Util/Compiler.h"

#include "GaussianMixture.h"
#include "EMInitializer.h"
#include "EMException.h"

int seed;
int ncl=1;
char *iname=(char *)"kmeans";
char *oname=nullptr;
char *fname=nullptr;


void ProcessArgs(int argc, char *argv[])
{
    int c;
    while ((c=getopt(argc,argv,"o:I:r:c:"))!=-1){
        switch(c) {

            case 'o':
                oname=optarg;
                break;
            case 'I':
                iname=optarg;
                break;
            case 'r':
                seed=atoi(optarg);
                if (seed<0) {
                    throw std::invalid_argument("invalid value of -r option\n");
                }
                break;
            case 'c':
                ncl=atoi(optarg);
                if (ncl<=0) {
                    throw std::invalid_argument("%invalid value of -c option\n");
                }
                break;

            default: throw std::invalid_argument("Invalid command line option");

        }
    }
    if (optind<argc)
        fname=argv[optind];
    else
        throw std::invalid_argument("Missing dataset name");
}





void InitRNGs() {
    printf("Seeed of rng: %d\n",seed);
#ifdef _OPENMP
#pragma omp parallel default(none)
    {
#pragma omp single
        printf("OpenMP version with %d threads\n",omp_get_num_threads());
    }
#else
    printf("Single threaded version\n");
#endif
    SRand(seed);
}

void DestroyRNGs() {
#pragma omp parallel default(none)
    {
        DelMTRand();
    }
}



int main(int argc, char *argv[]) {
    StandardStdOut::Init();
    Eigen::initParallel();
    char hostname[1024];
    gethostname(hostname, 1023);
    kma_printf("Running on machine: %s\n", hostname);
    try {
        ProcessArgs(argc,argv);
        InitRNGs();
        NUMAAllocator::CreateNUMAAllocator();
        CompileHeader(argv[0]);
        MultithreadedDataset D;
        kma_printf("Loading dataset %s... ",fname);
        D.Load(fname);
        kma_printf("%ld vectors, %d features\n",D.GetRowCount(),D.GetColCount());
        kma_printf("Covariance matrix trace: %10.10g\n",D.GetCovMatrixTrace());
        kma_printf("Number of components: %d\n",ncl);
        GaussianMixture G(ncl,D.GetColCount());
        EMInitializer *pInit=CreateEMInitializer(iname,D,ncl);
        JainEMInitializer Init(D,ncl);
        pInit->PrintInfo();
        pInit->Init(G);
        if (oname == nullptr)
            G.Dump();
        else {
            G.Write(oname);
        }
        DestroyRNGs();
    } catch (std::invalid_argument &E) {
        kma_printf("Invalid program argument: %s\n",E.what());
    }
}
