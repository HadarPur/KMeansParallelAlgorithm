#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "Allocation Header.h"

// Ensure that there are not a memory leak.
void checkAllocation(void* pointer)
{
	if (!pointer)
	{
		printf("Dynamic allocation failed\n");
		MPI_Finalize();
		exit(1);
	}
}

// Ensure that there are not a memory leak.
void freeDynamicAllocation(void* pointer)
{
	free(pointer);
}
