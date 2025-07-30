/*
 * MPIJainEMInitializer.h
 *
 *  Created on: Mar 27, 2017
 *      Author: wkwedlo
 */

#ifndef MPIEM2_MPIJAINEMINITIALIZER_H_
#define MPIEM2_MPIJAINEMINITIALIZER_H_

#include "../EM2/EMInitializer.h"
#include "../MPIKmeans/DistributedMultithreadedDataset.h"

class MPIJainEMInitializer: public EMInitializer {
	const char *fname;
public:
	MPIJainEMInitializer(DistributedMultithreadedDataset &D,int nC,const char *fn);
	virtual void Init(GaussianMixture &G);
	virtual ~MPIJainEMInitializer() {}
	virtual void PrintInfo() {}
};

#endif /* MPIEM2_MPIJAINEMINITIALIZER_H_ */
