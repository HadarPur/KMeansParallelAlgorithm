#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "KMeans Header.h"
#include "Allocation Header.h"
#include "File Header.h"
#include "MPI Type Decleration Header.h"

int main(int argc, char *argv[])
{
	Point* points;
	Cluster* clusters;
	Point** pointsMat; // group of points for each cluster
	int* clustersSize; // size for each group
	int i, errorCode = 999;
	int namelen, numprocs, myid, numOfPoints;
	int K;				// K - Number of clusters
	int N;				//  N - number of points
	double LIMIT;	// LIMIT - the maximum number of iterations for K - Mean algorithem.
	double QM;		// QM - quality measure to stop
	double T;			// T - defines the end of time interval 
	double dT;			// dT - defines the moments t = n*dT
	double quality, time;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;
	MPI_Comm grid_comm;

	if (numprocs < 3)
	{
		printf("Num of the proccess / computers must be grater than 3");
		MPI_Abort(MPI_COMM_WORLD, errorCode);
	}

	// MPI Point type
	MPI_Datatype PointType;
	createPointType(&PointType);

	// MPI Cluster type
	MPI_Datatype ClusterType;
	createClusterType(&ClusterType);

	// only the master read from 
	if (myid == MASTER)
	{
		// read points from file
		points = readDataFromFile(&N, &K, &LIMIT, &QM, &T, &dT);

		// each proccess will get numOfPoints to handle with beside the master
		numOfPoints = N / numprocs;

		// choose first K points as the initial clusters centers, Step 1 in K-Means algorithem
		clusters = initClusters(points, K);

		// send to all slaves the num of the clusters 
		MPI_Bcast(&K, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		// send to all slaves the numOfPoints that they need to handle with
		MPI_Bcast(&numOfPoints, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

		// the master handle with the rest of the points
		numOfPoints += (N % numprocs);
	}
	else
	{
		// all slaves recieve the num of the clusters 
		MPI_Bcast(&K, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		// all slaves recieve the numOfPoints that they need to handle with
		MPI_Bcast(&numOfPoints, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}

	// create matrix of points, where the number of rows are the number of the clusters.
	pointsMat = (Point**)calloc(K, sizeof(Point*));
	checkAllocation(pointsMat);
	//create array of integers, where the number of points for each clusters.
	clustersSize = (int*)calloc(K, sizeof(int));
	checkAllocation(clustersSize);

	//Start of the Algorithem
	quality = kMeansWithIntervals(points, clusters, pointsMat, clustersSize, N, K, LIMIT, QM, T, dT, &time);

	//Print the result of the K-Means algorithem -> the quality
	printf("The quality is : %lf\n", quality);

	// only the master wtite to the file
	if (myid == 0) 
	{
		//write final points from file
		writeToFile(time, quality, clusters, K);
	}

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
