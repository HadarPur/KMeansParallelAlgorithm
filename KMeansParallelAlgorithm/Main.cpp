#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "Header.h"

int main(int argc, char *argv[])
{
	int i;
	Point* points;
	Cluster* clusters;
	Point** pointsMat; // Each row I, contains the cluster I points.
	int* clustersSize; // Each array cell I contain the size of the row I in pointsMat.
	int totalNumOfPoints, K; // K - Number of clusters,limit - the maximum number of iterations for K-Mean algorithem.
	double QM, T, dt, quality, time, limit;

	int  namelen, numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;
	MPI_Comm grid_comm;

	//read points from file
	points = readDataFromFile(&totalNumOfPoints, &K, &limit, &QM, &T, &dt);
	//Choose first K points as the initial clusters centers, Step 1 in K-Means algorithem

	clusters = initClusters(points, K);

	//Create matrix of points, where the number of rows are the number of the clusters.
	pointsMat = (Point**)calloc(K, sizeof(Point*));
	checkAllocation(pointsMat);
	clustersSize = (int*)calloc(K, sizeof(int));
	checkAllocation(clustersSize);

	//Start of the Algorithem
	quality = kMeansWithIntervals(points, clusters, pointsMat, clustersSize, totalNumOfPoints, K, limit, QM, T, dt, &time);

	//Print the result of the K-Means algorithem -> the quality
	printf("The quality is : %lf\n", quality);

	writeToFile(time, quality, clusters, K);

	//Free memory from the heap (dynamic)
	freeDynamicAllocation(clustersSize);
	for (i = 0; i < K; i++)
	{
		freeDynamicAllocation(pointsMat[i]);
	}
	free(pointsMat);
	freeDynamicAllocation(clusters);
	freeDynamicAllocation(points);

	printf("bye bye\n");
	system("pause");

	MPI_Finalize();
	return 0;

}
