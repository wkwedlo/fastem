#include "Array.h"
#include "Debug.h"
#include <cstdio>
#include <numaif.h>
#include <numa.h>

void DebugHang();

void CheckArrayRange(long i,long Size)
{
    if (i<0 || i>= Size){
	TRACE2("Index %ld is out of range %ld\n",i,Size);
	DebugHang();
    }
}

extern inline int NodeOfAddr(void *ptr) {
	int Status[1];
	int ret=move_pages(0,1,&ptr,NULL,Status,0);
	return Status[0];
}


void NumaScan(char *ptr,long nBytes) {
	int PrevNode=-1;
	for(long i=0;i<nBytes;i++) {
		int Node=NodeOfAddr(ptr+i);
		if (PrevNode!=Node)
			printf("Byte %ld node %d\n",i,Node);
		PrevNode=Node;
	}
}
