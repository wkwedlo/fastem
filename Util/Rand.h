

#ifndef RAND_H
#define RAND_H


double Rand();
double RandNorm();
double RandCauchy();

int RandInt(const unsigned long n);

double RandDbl();
double RandDbl(const double n);

void SRand(unsigned seed);
void DelMTRand();

#endif

