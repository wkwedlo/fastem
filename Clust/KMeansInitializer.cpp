#include <limits>
#include <stdio.h>
#include "KMeansInitializer.h"
#include "../Util/Rand.h"
#include "../Util/FileException.h"

KMeansInitializer::KMeansInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl): 
					Data(D),CV(aCV),nclusters(cl) {
	ncols=Data.GetColCount();
}  




void KMeansInitializer::RepairCentroid(DynamicArray<OPTFLOAT> &vec,int clnum) {
	//printf("*");
	//fflush(stdout);
	
		const float *row=Data.GetRowNew((long)(Rand()*Data.GetRowCount()));
		for(int j=ncols*clnum,k=0;j<ncols*(clnum+1);j++,k++)
			vec[j]=row[k];
}



ForgyInitializer::ForgyInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl) : KMeansInitializer(D,aCV,cl) {
}

//#define TRACE_OBJNUMS
//#define TRACE_CENTROIDS


void ForgyInitializer::Init(DynamicArray<OPTFLOAT> &v) {
	ASSERT(v.GetSize()==ncols*nclusters);
	//printf("Using Forgy Initializer\n");
	for(int i=0;i<nclusters;i++) {
		long ObjNum=(long)(Rand()*Data.GetRowCount());
#ifdef TRACE_OBJNUMS
		TRACE1("Obj %d selected as initial centroid\n",ObjNum);
#endif
		const float * row=Data.GetRowNew(ObjNum);
		for(int j=0;j<ncols;j++)
			v[i*ncols+j]=row[j];
	}
#ifdef TRACE_CENTROIDS
	for(int i=0;i<nclusters*ncols;i++)
			TRACE2("v[%d]=%f\n",i,v[i]);
#endif
}




RandomInitializer::RandomInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl) : KMeansInitializer(D,aCV,cl) {
	PartCounts.SetSize(nclusters);
}

void RandomInitializer::Init(DynamicArray<OPTFLOAT> &v) {
	long nRows=Data.GetRowCount();
	
	for (int i=0;i<nclusters;i++)
		PartCounts[i]=0;
	
	for (int i=0;i<nclusters*ncols;i++)
		v[i]=(OPTFLOAT)0;
	
	for(long i=0;i<nRows;i++) {
		const float *row=Data.GetRowNew(i);
		int Part=(int)(Rand()*nclusters);
		PartCounts[Part]++;
		for(int j=0;j<ncols;j++)
			v[Part*ncols+j]+=row[j];
	}
	for (int i=0;i<nclusters;i++) {
		OPTFLOAT f=(OPTFLOAT)1.0/(OPTFLOAT)PartCounts[i];
		TRACE2("Cluster%d f=%5.3f\n",i,f);
		for (int j=0;j<ncols;j++)
			v[i*ncols+j]*=f;
	}
}



MinDistInitializer::MinDistInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl,int Tr) : KMeansInitializer(D,aCV,cl) {
	Trials=Tr;
}


void MinDistInitializer::Init(DynamicArray<OPTFLOAT> &v) {
    ASSERT(v.GetSize()==nclusters*ncols);
	DynamicArray<long> Nums(Trials);
	DynamicArray<OPTFLOAT> Center(ncols);
	long nRows=Data.GetRowCount();
	
	for(int i=0;i<ncols;i++) {
		Center[i]=0.0;
	}
	for(long i=0;i<nRows;i++) {
		const float *Row=Data.GetRowNew(i);
		for(int j=0;j<ncols;j++) {
			Center[j]+=Row[j];
		}
	}
	OPTFLOAT f=(OPTFLOAT)1.0/(OPTFLOAT)nRows;
	
	for(int i=0;i<ncols;i++) {
		Center[i]*=f;
	}
	
	OPTFLOAT MinDist=std::numeric_limits<OPTFLOAT>::max();
	for(int i=0;i<Trials;i++) {
		const float *Row=Data.GetRowNew((long)(Rand()*Data.GetRowCount()));
		OPTFLOAT Dist=CV.SquaredDistance(0,Center,Row);
		if (Dist<MinDist) {
			MinDist=Dist;
			for(int j=0;j<ncols;j++)
				v[j]=Row[j];
		}
	}
	
	
	/*DataRow &rfirst=Data.GetRow((int)(Rand()*nRows));
	for (int i=0;i<ncols;i++)
			v[i]=rfirst[i];*/
	for(int i=1;i<nclusters;i++) {
			for(int j=0;j<Trials;j++)
				Nums[j]=(long)(Rand()*Data.GetRowCount());
			double MaxDist=0;
			int Maxj=0;
			for(int j=0;j<Trials;j++) {
				const float *row=Data.GetRowNew(Nums[j]);
				double MinDist=std::numeric_limits<double>::max();
				for(int k=0;k<i;k++) {
					double SD=CV.SquaredDistance(k,v,row);
					//printf("v%d - c%d : %5.3f\n",j,k,SD);
					if (SD < MinDist)
						MinDist=SD;
				}
				//printf("v%d mindist %5.3f\n",j,MinDist);
				if (MinDist>MaxDist) {
					MaxDist=MinDist;
					Maxj=j;
				}
			}
			//printf("v%d selected\n",Maxj);
			const float *row=Data.GetRowNew(Nums[Maxj]);
			for(int j=ncols*i,k=0;j<ncols*(i+1);j++,k++)
				v[j]=row[k];
			}
	//exit(-1);
}

FileInitializer::FileInitializer(MultithreadedDataset &D,CentroidVector &aCV,int cl,const char *fname) : KMeansInitializer(D,aCV,cl) {
	FILE *pF=fopen(fname,"rb");
	if (pF==NULL)
		throw FileException("Cannot open file with centroid coordinates");
	int nCols,nRows;
	fread(&nRows,1,sizeof(nRows),pF);
	fread(&nCols,1,sizeof(nCols),pF);
	if (nCols!=ncols)
		throw FileException("Number of columns in dataset and centroid file do not match");
	if (nRows!=nclusters)
		throw FileException("Number of clusters in centroid file and -c value do not match");
	vec.SetSize(ncols*nclusters);
	if (fread(vec.GetData(),sizeof(OPTFLOAT),nclusters*ncols,pF)!=nclusters*ncols)
		throw FileException("fread from centroid file failed");
}

void FileInitializer::Init(DynamicArray<OPTFLOAT> &v) {
	v=vec;
}
