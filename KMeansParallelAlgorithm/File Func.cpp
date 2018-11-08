#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "KMeans Header.h"
#include "Allocation Header.h"

const char* FILE_PATH_INPUT = "D:\\KMeansParallelAlgorithm\\16Points.txt";
const char* FILE_PATH_OUTPUT = "D:\\KMeansParallelAlgorithm\\16PointsOutPut.txt";

// Read all the points from the file
Point* readDataFromFile(int* N, int* K, double* T, double* dt, double* limit, double* QM)
{
	int i;
	FILE* file = fopen(FILE_PATH_INPUT, "r");

	// Check if the file exist
	if (!file)
	{
		printf("could not open the file ");
		MPI_Finalize();
		exit(1);
	}
	// Getting the supplied data from input file
	fscanf(file, "%d %d %lf %lf %lf %lf\n", N, K, T, dt, limit, QM);

	Point* points = (Point*)malloc(*N * sizeof(Point));
	checkAllocation(points);

	// Initalize points from file
	for (i = 0; i < *N; i++)
	{
		fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &(points[i].x0), &(points[i].y0), &(points[i].z0), &(points[i].vx), &(points[i].vy), &(points[i].vz));
		points[i].x = 0;
		points[i].y = 0;
		points[i].z = 0;
		points[i].currentClusterIndex = 0;
		points[i].previousClusterIndex = -1;
	}

	fclose(file);
	return points;
}

// Write the results to the file
void writeToFile(double t, double q, Cluster* clusters, int K)
{
	FILE* file = fopen(FILE_PATH_OUTPUT, "w");

	if (file == NULL) {
		printf("Couldnt open the file\n");
		MPI_Finalize();
		exit(1);
	}

	fprintf(file, "First occurrence at t = %lf with q = %lf\nCenters of the clusters:\n", t, q);

	for (int i = 0; i < K; i++)
	{
		fprintf(file, "X = %-15lf\tY = %-15lf\tZ = %-15lf\n", clusters[i].x, clusters[i].y, clusters[i].z);
	}
	fclose(file);
}
