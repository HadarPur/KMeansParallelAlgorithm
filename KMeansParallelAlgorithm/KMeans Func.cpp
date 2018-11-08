#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "KMeans Header.h"
#include "Allocation Header.h"

// Step 1 in K-Means algorithm
Cluster* initClusters(const Point* points, int K)
{
	int i;
	Cluster* clusters = (Cluster*)malloc(K * sizeof(Cluster));
	checkAllocation(clusters);

	for (i = 0; i < K; i++)
	{
		clusters[i].x = points[i].x0;
		clusters[i].y = points[i].y0;
		clusters[i].z = points[i].z0;
		clusters[i].diameter = 0;
	}
	return clusters;
}

// initialize the help arrays 
void initPointsInfoArray(double* initialPointsCor, double* currentPointsCor, double* velocityPointsCor, Point* points ,int numOfPoints)
{
	for (int i = 0; i < numOfPoints; i++)
	{
		initialPointsCor[i * NUM_OF_DIMENSIONS] = points[i].x0;
		initialPointsCor[(i * NUM_OF_DIMENSIONS)+1] = points[i].y0;
		initialPointsCor[(i * NUM_OF_DIMENSIONS)+2] = points[i].z0;

		currentPointsCor[i * NUM_OF_DIMENSIONS] = points[i].x;
		currentPointsCor[(i * NUM_OF_DIMENSIONS) + 1] = points[i].y;
		currentPointsCor[(i * NUM_OF_DIMENSIONS) + 2] = points[i].z;

		velocityPointsCor[i * NUM_OF_DIMENSIONS] = points[i].vx;
		velocityPointsCor[(i * NUM_OF_DIMENSIONS) + 1] = points[i].vy;
		velocityPointsCor[(i * NUM_OF_DIMENSIONS) + 2] = points[i].vz;
	}
}

// Part of step 2 in K-Means algorithm
int getClosestClusterIndex(double x, double y, double z, Cluster* clusters, int K)
{
	int i, index = 0;
	double minDistance, tempDistance;

	minDistance = distancePointToPoint(x, y, z, clusters[0].x, clusters[0].y, clusters[0].z);

	for (i = 1; i < K; i++)
	{
		tempDistance = distancePointToPoint(x, y, z, clusters[i].x, clusters[i].y, clusters[i].z);
		if (tempDistance < minDistance)
		{
			minDistance = tempDistance;
			index = i;
		}
	}
	return index;
}

// Group points around the given cluster centers (2)
void groupPointsToClusters(Point** pointsMat, int* clustersSize, Point* points, int totalNumOfPoints, Cluster* clusters, int K)//Step 2 in K-Means algorithm
{
	int i;

	//Reset ClustersSize Array Cells
	for (i = 0; i < K; i++)
	{
		clustersSize[i] = 0;
	}

	//finding for each point his closet cluster
	for (i = 0; i < totalNumOfPoints; i++)
	{
		points[i].previousClusterIndex = points[i].currentClusterIndex;
		points[i].currentClusterIndex = getClosestClusterIndex(points[i].x, points[i].y, points[i].z, clusters, K);
	}

	for (i = 0; i < totalNumOfPoints; i++)
	{
		clustersSize[points[i].currentClusterIndex]++;
		pointsMat[points[i].currentClusterIndex] = (Point*)realloc(pointsMat[points[i].currentClusterIndex], clustersSize[points[i].currentClusterIndex] * sizeof(Point));
		pointsMat[points[i].currentClusterIndex][(clustersSize[points[i].currentClusterIndex]) - 1] = points[i];
	}
}

// Calculate distance between 2 points
double distancePointToPoint(double x1, double y1, double z1, double x2, double y2, double z2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2) + pow((z1 - z2), 2));
}


// Calculate point position by time
void calPointsCordinates(Point* points, int N, double t)
{
	int i;
	for (i = 0; i < N; i++)
	{
		points[i].x = points[i].x0 + (t*points[i].vx);
		points[i].y = points[i].y0 + (t*points[i].vy);
		points[i].z = points[i].z0 + (t*points[i].vz);
	}
}

// Recalculate the cluster centers - step 3 in K-Means algorithm
void calSumPointsCorToCluster(Point* clusterPoints, int clusterPointsSize, double* sumX, double* sumY, double* sumZ)
{
	*sumX = *sumY = *sumZ = 0;

	// Calculate all the cluster points cordinates
	for (int i = 0; i < clusterPointsSize; i++)
	{
		*sumX += clusterPoints[i].x;
		*sumY += clusterPoints[i].y;
		*sumZ += clusterPoints[i].z;
	}
}

// Calculate diameters for specific cluster
double calClusterDiameter(Point* clusterPoints, int clusterPointsSize)
{
	int i, j;
	double maxDistance = 0, tempDistance = 0;
	for (i = 0; i < clusterPointsSize; i++)
	{
		for (j = i + 1; j < clusterPointsSize; j++)
		{
			tempDistance = distancePointToPoint(clusterPoints[i].x, clusterPoints[i].y, clusterPoints[i].z, clusterPoints[j].x, clusterPoints[j].y, clusterPoints[j].z);

			if (maxDistance < tempDistance)
			{
				maxDistance = tempDistance;
			}
		}
	}
	return maxDistance;
}

// Evaluate the quality of the clusters found (6)
double evaluateQuality(Point** pointsMat, Cluster* clusters, int K, int* clustersSize)
{
	int i, j;
	double numerator = 0, quality = 0, numOfArguments, currentClustersDistance = 0;

	numOfArguments = K * (K - 1);

	for (i = 0; i < K; i++)
	{
		// Calculate the current cluster's diameter (di) 
		clusters[i].diameter = calClusterDiameter(pointsMat[i], clustersSize[i]);

		for (j = 0; j < K; j++)
		{
			if (i != j)
			{
				// Calculate the distance between the current cluster and the other clusters (Dij)
				currentClustersDistance = distancePointToPoint(clusters[i].x, clusters[i].y, clusters[i].z, clusters[j].x, clusters[j].y, clusters[j].z);

				numerator += clusters[i].diameter / currentClustersDistance;
			}
		}
	}

	// Calculate the average of diameters of the cluster divided by distance to other clusters
	quality = numerator / numOfArguments;
	return quality;
}

// KMeans algorithm with x,y that changing by time
double kMeansWithIntervalsForMaster(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K, double limit, double QM, double T, double dt, double* time,
	MPI_Datatype PointType, MPI_Datatype ClusterType, int numprocs, double* initialPointsCor, double* currentPointsCor, double* velocityPointsCor, double* sumPointsCenters)
{
	double n, tempQuality=0, quality = 0;
	// Match points to clusters

	for (*time = 0, n = 0; n <= T / dt; n++)
	{
		// Calculate the current time
		*time = n*dt;

		printf("t = %lf Quality = %lf\n", *time, quality);
		fflush(stdout);

		// Calculate points cordinates according to current time
		calPointsCordinates(points, numOfPoints, *time);

		//sends to each slave the relevante time
		for (int proccessID = 1; proccessID < numprocs; proccessID++)
		{
			MPI_Send(time, 1, MPI_DOUBLE, proccessID, TRANSFER_TAG, MPI_COMM_WORLD); 
		}

		// K-Mean Algorithm
		tempQuality = kMeansAlgorithmMaster(points, clusters, pointsMat, clustersSize, numOfPoints, K, limit, numprocs, PointType, ClusterType, sumPointsCenters);


		// Checks if the quality measure is less than QM
		if (tempQuality < QM)
		{
			//sends to each slave the to finish the algorithm
			for (int proccessID = 1; proccessID < numprocs; proccessID++)
			{
				MPI_Send(time, 1, MPI_DOUBLE, proccessID, FINAL_TAG, MPI_COMM_WORLD); 
			}
			return tempQuality;
		}

		// Checks if the current given quality measure is better than the best given quality so far.
		if (tempQuality < quality || quality == 0)
			quality = tempQuality;
	}

	//sends to each slave the to finish the algorithm
	for (int proccessID = 1; proccessID < numprocs; proccessID++)
	{
		MPI_Send(time, 1, MPI_DOUBLE, proccessID, FINAL_TAG, MPI_COMM_WORLD); 
	}

	return quality;
}

// KMeans algorithm with x,y that changing by time for slave operation
void kMeansWithIntervalsForSlave(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	MPI_Datatype PointType, MPI_Datatype ClusterType, double* initialPointsCor, double* currentPointsCor, double* velocityPointsCor, double* sumPointsCenters)
{
	double time;

	MPI_Status status;
	status.MPI_TAG = TRANSFER_TAG;

	while (status.MPI_TAG != FINAL_TAG)
	{
		MPI_Recv(&time, 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG != FINAL_TAG)
		{
			// Calculate points cordinates according to current time
			calPointsCordinates(points, numOfPoints, time);

			// slave the kmeans algorithm
			kMeansAlgorithmSlave(points, clusters, pointsMat, clustersSize, numOfPoints, K, PointType, ClusterType, sumPointsCenters);
		}
		else
		{
			break;
		}
	}
}

// K-Means algorithm - Master
double kMeansAlgorithmMaster(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K, int limit, int numOfProccess, MPI_Datatype PointType, MPI_Datatype ClusterType, double* sumPointsCenters)
{
	MPI_Status status;

	double* totalSumPointsCenters;
	int* totalClusterSize;
	int j, flag, totalFlag;

	// initial arrays or the final answer
	totalSumPointsCenters = (double*)calloc(K*NUM_OF_DIMENSIONS, sizeof(double));
	totalClusterSize = (int*)calloc(K, sizeof(int));

	reinitializePreviousClusterIndex(points, numOfPoints);

	for (int i = 0; i < limit; i++)
	{
		flag = 0;
		totalFlag = 0;

		//sends to each slave the clusters
		for (int proccessID = 1; proccessID < numOfProccess; proccessID++)
		{
			MPI_Send(clusters, K, ClusterType, proccessID, TRANSFER_TAG, MPI_COMM_WORLD);
		}

		// Step 2 - Group points around the given clusters centers
		groupPointsToClusters(pointsMat, clustersSize, points, numOfPoints, clusters, K);

		// Step 3 - Recalculate the clusters center (in the parallel code each proccess calculate the sum of coordinate x,y,z)
		for (j = 0; j < K; j++)
		{
			calSumPointsCorToCluster(pointsMat[j], clustersSize[j], sumPointsCenters + (j * NUM_OF_DIMENSIONS), sumPointsCenters + ((j * NUM_OF_DIMENSIONS) + 1), sumPointsCenters + ((j * NUM_OF_DIMENSIONS) + 2));
		}

		// get the info from all the slaves to calculate the total answers
		MPI_Reduce(clustersSize, totalClusterSize, K, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
		MPI_Reduce(sumPointsCenters, totalSumPointsCenters, K*NUM_OF_DIMENSIONS, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

		// update the cluster centers
		recalculateClusterCenters(clusters, K, totalSumPointsCenters, totalClusterSize);

		// Step 4 - Checks if some point move to another cluster after the update of clusetrs center cordinates.
		for (j = 0; j < numOfPoints && (points[j].currentClusterIndex == points[j].previousClusterIndex); j++);
		flag = (j == numOfPoints ? 1 : 0);

		// each proccess send to the master the answer, and like that we will know if there is a movement between the clusters
		MPI_Reduce(&flag, &totalFlag, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

		// Step 5 - check the terminition condition
		if (totalFlag == numOfProccess)
		{
			//sends to each slave to finish
			for (int proccessID = 1; proccessID < numOfProccess; proccessID++)
			{
				MPI_Send(clusters, K, ClusterType, proccessID, MID_TAG, MPI_COMM_WORLD);
			}

			// need to gather
			gatherThePoints(pointsMat, clustersSize, totalClusterSize, K, numOfProccess, PointType);

			freeDynamicAllocation(totalClusterSize);
			freeDynamicAllocation(totalSumPointsCenters);
			return evaluateQuality(pointsMat, clusters, K, clustersSize);
		}
	}

	//sends to each slave to finish
	for (int proccessID = 1; proccessID < numOfProccess; proccessID++)
	{
		MPI_Send(clusters, K, ClusterType, proccessID, MID_TAG, MPI_COMM_WORLD);
	}

	// need to gather
	gatherThePoints(pointsMat, clustersSize, totalClusterSize, K, numOfProccess, PointType);

	freeDynamicAllocation(totalClusterSize);
	freeDynamicAllocation(totalSumPointsCenters);

	return evaluateQuality(pointsMat, clusters, K, clustersSize);
}

// K-Means algorithm - Slave
void kMeansAlgorithmSlave(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K, MPI_Datatype PointType, MPI_Datatype ClusterType, double* sumPointsCenters)
{
	int j, flag;
	MPI_Status status;
	status.MPI_TAG = TRANSFER_TAG;

	reinitializePreviousClusterIndex(points, numOfPoints);

	while (status.MPI_TAG == TRANSFER_TAG)
	{
		flag = 0;
		MPI_Recv(clusters, K, ClusterType, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == MID_TAG) // kmeans finished for the current time
		{
			// return the points to the master
			for (int i = 0; i < K; i++)
			{
				MPI_Send(&(clustersSize[i]), 1, MPI_INT, MASTER, TAG, MPI_COMM_WORLD);
				MPI_Send(&(pointsMat[i]), clustersSize[i], PointType, MASTER, TAG, MPI_COMM_WORLD);
			}
		}
		else // we need to continue te algorithm kmeans
		{
			// Step 2 - Group points around the given clusters centers
			groupPointsToClusters(pointsMat, clustersSize, points, numOfPoints, clusters, K);

			// Step 3 - Recalculate the clusters center (in the parallel code each proccess calculate the sum of coordinate x,y,z)
			for (j = 0; j < K; j++)
			{
				calSumPointsCorToCluster(pointsMat[j], clustersSize[j], sumPointsCenters + (j * NUM_OF_DIMENSIONS), sumPointsCenters + (j * NUM_OF_DIMENSIONS)+1, sumPointsCenters + (j * NUM_OF_DIMENSIONS)+2);
			}

			MPI_Reduce(clustersSize, clustersSize, K, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
			MPI_Reduce(sumPointsCenters, sumPointsCenters, K*NUM_OF_DIMENSIONS, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

			// Step 4 - Checks if some point move to another cluster after the update of clusetrs center cordinates.
			for (j = 0; j < numOfPoints && (points[j].currentClusterIndex == points[j].previousClusterIndex); j++);
			flag = (j == numOfPoints ? 1 : 0);

			MPI_Reduce(&flag, &flag, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
		}
	}
}

// reinitialize the value  to be -1
void reinitializePreviousClusterIndex(Point* points, int numOfPoints)
{
	for (int i = 0; i < numOfPoints; i++)
	{
		points[i].previousClusterIndex = -1;
	}
}

// after gets the all info from the slave, recalculate the cluster centes again
void recalculateClusterCenters(Cluster* clusters, int K, double* totalSumPointsCenters, int* totalClusterSize)
{
	for (int i = 0; i < K; i++)
	{
		clusters[i].x = totalSumPointsCenters[(i*NUM_OF_DIMENSIONS)] / totalClusterSize[i];
		clusters[i].y = totalSumPointsCenters[(i*NUM_OF_DIMENSIONS)+1] / totalClusterSize[i];
		clusters[i].z = totalSumPointsCenters[(i*NUM_OF_DIMENSIONS)+2] / totalClusterSize[i];
	}
}

void gatherThePoints(Point** pointsMat, int* clustersSize, int* totalClusterSize, int K, int numOfProccess, MPI_Datatype PointType)
{
	MPI_Status status;
	int prevSize, tmpSize;

	for (int i = 0; i < K; i++)
	{
		pointsMat[i] = (Point*)realloc(pointsMat[i], totalClusterSize[i] * (sizeof(Point)));
		//gets from each slave the data
		for (int proccessID = 1; proccessID < numOfProccess; proccessID++)
		{
			prevSize = clustersSize[i];
			MPI_Recv(&tmpSize, 1, MPI_INT, proccessID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			clustersSize[i] += tmpSize;
			MPI_Recv(pointsMat[i]+prevSize, tmpSize, PointType, proccessID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}
}

