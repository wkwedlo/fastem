#include "Rand.h"
#include <random>

using namespace std;


mt19937_64 rng;
uniform_real_distribution<> uniform;
normal_distribution<> norm;

double Rand() 
{
	return uniform(rng);
}


double RandCauchy()
{
	return tan(M_PI*(Rand()-0.5));
}

double RandNorm()
{
	return norm(rng);
}


int RandInt(const unsigned long n) {
	return Rand()*n;
}


void SRand(unsigned seed) 
{
	rng.seed(seed);
}

void DelMTRand() {
}

