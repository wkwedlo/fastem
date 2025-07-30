/*
 * EMArgs.h
 *
 *  Created on: Sep 16, 2010
 *      Author: wkwedlo
 */

#ifndef EMARGS_H_
#define EMARGS_H_


int ProcessArgs(int argc, char *argv[]);


extern char *fname;
extern char *oname;
extern char *cname;
extern char *iname;
extern char *aname;
extern char *rname;
extern char *mname;
extern char *affname;
extern char *numaname;
extern char *numaparam;


extern int seed;
extern int ncl;
extern int verbosity;
extern char *init;
extern int maxiter;
extern int burnin;
extern double runtime;
extern double Eps;
extern double athr;
extern double regcoeff;
extern bool svd;
extern double hugepagespernode;
extern char *msteploopparam;
extern bool benchmarkreducer;

#endif /* EMARGS_H_ */
