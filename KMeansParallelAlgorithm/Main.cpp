#include "KMeans Header.h"
#include "Allocation Header.h"
#include "File Header.h"
#include "MPI Type Decleration Header.h"

int main(int argc, char *argv[])
{
	Point* points;
	Cluster* clusters;

	Point** pointsMat; // group of points for each cluster
	int* clustersSize; // size for each group, size for each row 

	double* sumPointsCenters;

	int i, errorCode = 999;
	int namelen, numprocs, myid, numOfPoints;
	int K;				// K - Number of clusters
	int N;				//  N - number of points
	double LIMIT;	// LIMIT - the maximum number of iterations for K - Mean algorithem.
	double QM;		// QM - quality measure to stop
	double T;			// T - defines the end of time interval 
	double dT;			// dT - defines the moments t = n*dT
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;
	MPI_Comm grid_comm;

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
		points = readDataFromFile(&N, &K, &T, &dT, &LIMIT, &QM);

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
		// all slaves recieve the numOfPoints they need to handle with
		MPI_Bcast(&numOfPoints, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	}

	// create matrix of points, where the number of rows are the number of the clusters.
	pointsMat = (Point**)calloc(K, sizeof(Point*));
	checkAllocation(pointsMat);

	// create array of integers, where the number of points for each clusters.
	clustersSize = (int*)calloc(K, sizeof(int));
	checkAllocation(clustersSize);

	// sum array to calculate the average x,y,z 
	sumPointsCenters = (double*)calloc(K * NUM_OF_DIMENSIONS, sizeof(double));
	checkAllocation(sumPointsCenters);

	if (myid == MASTER)
	{
		double start, end; // to measure time
		double quality, time;

		int numOfPointForEachProc = numOfPoints - (N % numprocs);
		//sends to each slave the relevante points
		for (int proccessID = 1; proccessID < numprocs; proccessID++)
		{
			MPI_Send(points + numOfPoints+ (numOfPointForEachProc * (proccessID-1)), numOfPointForEachProc, PointType, proccessID, TAG, MPI_COMM_WORLD);
		}

		printf("K-Means Algorithm status: start\n");
		fflush(stdout);

		start = omp_get_wtime();

		//	master start the algorithm with the intervals
		quality = kMeansWithIntervalsForMaster(points, clusters, pointsMat, clustersSize, numOfPoints, K, LIMIT, QM, T, dT, &time, PointType, ClusterType, numprocs, sumPointsCenters);

		end = omp_get_wtime();

		//write final points from file
		writeToFile(time, quality, clusters, K);

		printf("\nK-Means Algorithm status: finish after %lf sec, with quality of %lf\nAlso the output file is ready\n\n", end - start, quality);
		fflush(stdout);
	}
	else
	{
		// allocation points array for each salve
		points = (Point*)calloc(numOfPoints, sizeof(Point));
		checkAllocation(points);
		// allocation points array for each salve
		clusters = (Cluster*)calloc(K, sizeof(Cluster));
		checkAllocation(clusters);

		MPI_Recv(points, numOfPoints, PointType, MASTER, TAG, MPI_COMM_WORLD, &status);

		// slave start the algorithm with the time that get from the master
		kMeansWithIntervalsForSlave(points, clusters, pointsMat, clustersSize, numOfPoints, K, PointType, ClusterType, sumPointsCenters);
	}

	//Free memory from the heap (dynamic)
	freeDynamicAllocation(clustersSize);
	for (i = 0; i < K; i++)
	{
		freeDynamicAllocation(pointsMat[i]);
	}
	freeDynamicAllocation(pointsMat);
	freeDynamicAllocation(clusters);
	freeDynamicAllocation(points);
	freeDynamicAllocation(sumPointsCenters);

	printf("The proccess %d is finished with computer name: %s\n", myid, processor_name);
	fflush(stdout);

	MPI_Finalize();
	return 0;
}
