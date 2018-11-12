#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define BLOCK_SIZE 1024

int getMaxThreadPerBlock();
int calPointsCoordinatesWithCuda(double time, double* initPointsCordinates, double* pointsVelocityArr, double* currentPointsCordniates, int size);
cudaError_t computePointsCordinates(double time, double* initPointsCordinates, double* pointsVelocityArr, double* currentPointsCordniates, int size);
__global__ void pointsMovementCalKernel(int size, double* dev_initPointsCordinates, double* dev_pointsVelocityArr, double* dev_currentPointsCordinates, double time);
void error(double* dev_currentPointsCordinates, double* dev_pointsVelocityArr, double* dev_initPointsCordinates);
