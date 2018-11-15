#include "KMeans Header.h"
#include <stdio.h>

__global__ void callPointsCoordinatesByTimeWithCuda(Point* points, int numOfPoints, double time)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadIndex = bid*CUDA_BLOCK_SIZE + tid;

	if (threadIndex < numOfPoints)
	{
		points[threadIndex].x = points[threadIndex].x0 + points[threadIndex].vx * time;
		points[threadIndex].y = points[threadIndex].y0 + points[threadIndex].vy * time;
		points[threadIndex].z = points[threadIndex].z0 + points[threadIndex].vz * time;
	}
}

void callPointsCoordinatesWithCuda(Point* points, int numOfPoints, double time)
{
	int numOfBlocks;
	Point* device_points;
	cudaError_t cudaStatus;

	//numOfBlocks = get_block_num(numOfPoints);
	numOfBlocks = numOfPoints / CUDA_BLOCK_SIZE;
	if (numOfPoints % CUDA_BLOCK_SIZE > 0)
		numOfBlocks++;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cudaFree(device_points);
	}

	// Allocate memory on GPU 
	cudaStatus = cudaMalloc((void**)&device_points, numOfPoints * sizeof(Point));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc dpPoints failed!");
		cudaFree(device_points);
	}

	// Copy memory from CPU to GPU
	cudaStatus = cudaMemcpy(device_points, points, numOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy dpPoints failed!");
		cudaFree(device_points);
	}

	// kernel function,each thread gets: dpPoints, part_size, dT
	callPointsCoordinatesByTimeWithCuda << <numOfBlocks, CUDA_BLOCK_SIZE >> >(device_points, numOfPoints, time);

	// Check errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "perform_deltaT_movements launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(device_points);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns errors
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching perform_deltaT_movements!\n", cudaStatus);
		cudaFree(device_points);
	}

	// Copy memory from GPU to CPU memory
	cudaStatus = cudaMemcpy(points, device_points, numOfPoints * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy from device  failed!");
		cudaFree(device_points);
	}
	cudaFree(device_points);
}