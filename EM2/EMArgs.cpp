/*
 * EMArgs.cpp
 *
 *  Created on: Sep 16, 2010
 *      Author: wkwedlo
 */
#include <stdexcept>
#include <string>

#include <cstdio>
#include <unistd.h>
#include <cstdlib>
#include "EMArgs.h"

using namespace std;

char *fname;
char *oname=NULL;
char *cname=NULL;
char *iname=NULL;
char *aname=NULL;
char *rname=NULL;
char *mname=NULL;
char *affname=NULL;
char *msteploopparam=nullptr;

char *numaname=nullptr;
char *numaparam=nullptr;

int seed;
int ncl=1;
int verbosity=1;
char *init=NULL;
int maxiter=1000000;
int burnin=0;
double regcoeff=0.0;
double runtime=0.0;
double Eps=1e-5;
double athr=1e5;
bool svd=true;
bool benchmarkreducer=false;



void ExitUsage() {
	printf("Arguments: filename - name of the file in .bin format\n");
	printf("-o filename : writes components in a binary format to the file filename\n");
	printf("-O filename : writes classes in a binary format to the file filename\n");
	printf("-r seed : sets the seed of pseudo-random number generator\n");
	printf("-c components : selects the number of Gaussian mixture components\n");
	printf("-t init : selects the method of initialization of mixture, choices are simple,km,kmper,jain\n");
	printf("-m emiter : sets the maximal number of EM iterations\n");
	printf("-b burnin : sets the number of burn-in EM iterations (default is 0)\n");
	printf("-f eps : sets the Eps EM stopping threshold (default 1e-5)\n");
	printf("-a abort: set EM failure threshold (default is 1.0e5)\n");
	printf("-A algorithm: selects a version of EM algorithm\n");
	printf("-R reducer: selects a version of OpenMP reduction algorithm\n");
	printf("-M msteploop: selects a version of FastEM M-step outer loop\n");
	printf("-N msteploopparam: sets aparameter values for FastEM M-step outer loop\n");
	printf("-B benchmarks reduction algorithm (-m sets the number of iterations -N number of columns)\n");
    printf("-T fname outputs thread placement info to file fname\n");
    printf("-u alloc use NUMA allocator alloc\n");
    printf("-U allocparam use NUMA allocator parameter allocparam\n");
    printf("-g coeff use regularization coefficient for covariance matrices, default 0.0\n");

	printf("-S disables svd\n");
	printf("-v verbosity\n");

}


int ProcessArgs(int argc, char *argv[])
{
	int c;
	while ((c=getopt(argc,argv,"BN:M:o:O:r:c:t:m:f:a:A:Sv:R:T:I:u:U:b:g:"))!=-1){
		switch(c) {
        case 'u':
            numaname=optarg;
            break;
        case 'U':
            numaparam=optarg;
            break;

        case 'B' :
					benchmarkreducer=true;
					break;

		case 'A':
					aname=optarg;
					break;
		case 'R':
					rname=optarg;
					break;
		case 'T':
					affname=optarg;
					break;
		case 'M':
					mname=optarg;
					break;
		case 'S':
					svd=false;
					break;
			case 'o':
					oname=optarg;
					break;
			case 'I':
					iname=optarg;
					break;
			case 'O':
					cname=optarg;
                break;

			case 'r':
				seed=atoi(optarg);
			if (seed<0) {
				throw std::invalid_argument("invalid value of -r option\n");
			}
			break;
			case 'b':
					burnin=atoi(optarg);
					if (burnin<0) {
						throw std::invalid_argument("invalid value of -b option\n");
					}
			break;
			case 'N':
					msteploopparam=optarg;
					break;
			case 'c':
					ncl=atoi(optarg);
					if (ncl<=0) {
						throw std::invalid_argument("%invalid value of -c option\n");
					}
					break;
			case 't':
					init=optarg;
					break;

			case 'm':
					maxiter=atoi(optarg);
					if (maxiter<0) {
						throw std::invalid_argument("invalid value of -m option\n");
					}
					break;
			case 'v':
					verbosity=atoi(optarg);
					if (verbosity<0) {
						throw std::invalid_argument("invalid value of -v option\n");
					}
					break;
			case 'a':
					athr=atof(optarg);
					if (athr<1.0 ) {
						throw std::invalid_argument("argument to -a must be greater then 1.0\n");
					}
					break;
            case 'f':
                Eps=atof(optarg);
                if (Eps<=0.0) {
                    throw std::invalid_argument("argument to -f must be greater or than 0\n");
                    return -1;
                }
                break;
            case 'g':
                regcoeff=atof(optarg);
                if (regcoeff<0.0 || regcoeff>1.0) {
                    throw std::invalid_argument("argument to -g must be small positive number\n");
                    return -1;
                }
                break;

			default: throw std::invalid_argument("Invalid command line option");

		}
	}
	if (optind<argc)
		fname=argv[optind];
	else if (!benchmarkreducer){
		ExitUsage();
		return -1;
	}
	return 1;


}
