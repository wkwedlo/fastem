#include "KMeansWithoutMSE.h"
#include "../../Util/Debug.h"
#include "../../Util/StdOut.h"
#include "../../Util/PrecisionTimer.h"
#include <time.h>

KMeansWithoutMSE::KMeansWithoutMSE(CentroidVector &CV, MultithreadedDataset &Data, int &IterCount)
	: cv(CV), data(Data), iterCount(IterCount) {
}

void KMeansWithoutMSE::RunKMeansWithoutMSE(DynamicArray<OPTFLOAT> &vec, const int verbosity,int MaxIter) {
	ASSERT(vec.GetSize() == cv.GetNCols() * cv.GetNClusters());

	long distanceCount = 0;
	double distanceCalculationRatio = 0;
	double distanceCalculationRatioSum = 0;
	PrecisionTimer T(CLOCK_MONOTONIC);
	InitDataStructures(vec);
	if (verbosity>0) kma_printf("InitDataStructures took %g seconds\n",T.GetTimeDiff());
	int i = 0;
	for(;;i++) {
		distanceCount = 0;

		bool cont = CorrectWithoutMSE(vec, &distanceCount);
	  	double iterTime=T.GetTimeDiff();

		distanceCalculationRatio = (double) (distanceCount / ((double) (data.GetTotalRowCount() * cv.GetNClusters()))) * 100;
		distanceCalculationRatioSum += distanceCalculationRatio;

		if (verbosity > 3) {
			kma_printf("i: %d, dcr: %g, itime: %g seconds\n", i, distanceCalculationRatio, iterTime);
		}


		if (i > 0 && !cont) {
			break;
		}
		if (MaxIter>0 && i==MaxIter-1)
			break;
	}

	iterCount += i + 1;

	if (verbosity > 0) {
		kma_printf("Avg distance calculation ratio: %g\n", distanceCalculationRatioSum / iterCount);
	}
}
