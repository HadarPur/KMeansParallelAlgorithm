#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "KMeans Header.h"
#include "Allocation Header.h"

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

// Recalculate the cluster centers - step 3 in K-Means algorithm
void calClusterCenter(Cluster* cluster, Point* clusterPoints, int clusterPointsSize)
{
	int i;
	double sumX = 0, sumY = 0, sumZ = 0;

	// Calculate all the cluster points cordinates
	for (int i = 0; i < clusterPointsSize; i++)
	{
		sumX += clusterPoints[i].x;
		sumY += clusterPoints[i].y;
		sumZ += clusterPoints[i].z;
	}

	// Finding the new cluster cordinates(center).
	cluster->x = (sumX / clusterPointsSize);
	cluster->y = (sumY / clusterPointsSize);
	cluster->z = (sumZ / clusterPointsSize);
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
double kMeansWithIntervals(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int totalNumOfPoints, int K, double limit, double QM, double T, double dt, double* time)
{
	int i;
	double n, tempQuality, quality = 0;
	// Match points to clusters

	for (*time = 0, n = 0; n <= T / dt; n++)
	{
		// Calculate the current time
		*time = n*dt;

		printf("t = %lf\n", *time);
		fflush(stdout);

		// Calculate points cordinates according to current time
		calPointsCordinates(points, totalNumOfPoints, *time);

		// K-Mean Algorithm
		tempQuality = kMeansAlgorithem(points, clusters, pointsMat, clustersSize, totalNumOfPoints, K, limit);

		// Checks if the quality measure is less than QM
		if (tempQuality < QM)
			return tempQuality;

		// Checks if the current given quality measure is better than the best given quality so far.
		if (tempQuality < quality || quality == 0)
			quality = tempQuality;
	}
	return quality;
}

// K-Means algorithm
double kMeansAlgorithem(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int totalNumOfPoints, int K, double limit)
{
	int i, j;

	for (i = 0; i < limit; i++)
	{
		// Step 2 - Group points around the given clusters centers
		groupPointsToClusters(pointsMat, clustersSize, points, totalNumOfPoints, clusters, K);

		// Step 3 - Recalculate the clusters center
		for (j = 0; j < K; j++)
		{
			calClusterCenter(clusters + j, pointsMat[j], clustersSize[j]);
		}

		// Step 4 - Checks if some point move to another cluster after the update of clusetrs center cordinates.
		for (j = 0; j < totalNumOfPoints && (points[j].currentClusterIndex == points[j].previousClusterIndex); j++);
		if (j == totalNumOfPoints)
			break;
	}
	return evaluateQuality(pointsMat, clusters, K, clustersSize);
}
