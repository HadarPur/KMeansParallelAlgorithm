#define _CRT_SECURE_NO_WARNINGS

#define MASTER 0
#define TAG 0
#define TRANSFER_TAG 0
#define MID_TAG 1
#define FINAL_TAG 2
#define NUM_OF_DIMENSIONS 3
#define POINT_STRUCT_SIZE 11
#define CLUSTER_STRUCT_SIZE 4

// Structure for clusters
typedef struct Cluster
{
	double x;
	double y;
	double z;
	double diameter; //The largest distance between 2 points in the current cluster
};

// Structure for each point
typedef struct Point
{
	double x0;
	double y0;
	double z0;
	double x;
	double y;
	double z;
	double vx;
	double vy;
	double vz;
	int currentClusterIndex;
	int previousClusterIndex;
};

Cluster* initClusters(const Point* points, int K);
int getClosestClusterIndex(double x, double y, double z, Cluster* clusters, int K);
void groupPointsToClusters(Point** pointsMat, int* clustersSize, Point* points, int N, Cluster* clusters, int K);
double distancePointToPoint(double x1, double y1, double z1, double x2, double y2, double z2);
void calSumPointsCorToCluster(Point* clusterPoints, int clusterPointsSize, double* sumX, double* sumY, double* sumZ);
double evaluateQuality(Point** pointsMat, Cluster* clusters, int K, int* clustersSize);
double calClusterDiameter(Point* clusterPoints, int clusterPointsSize);
void calPointsCoordinates(Point* points, int totalNumOfPoints, double t);
double kMeansWithIntervalsForMaster(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K, double limit, double QM, double T, double dt, double* time,
	MPI_Datatype PointType, MPI_Datatype ClusterType, int numprocs, double* initialPointsCoordinates, double* currentPointsCoordinates, double* velocityPointsCoordinates, double* sumPointsCenters);
void initPointsInfoArray(double* initialPointsCoordinates, double* currentPointsCoordinates, double* velocityPointsCoordinates, Point* points, int numOfPoints);
void kMeansWithIntervalsForSlave(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	MPI_Datatype PointType, MPI_Datatype ClusterType, double* initialPointsCoordinates, double* currentPointsCoordinates, double* velocityPointsCoordinates, double* sumPointsCenters);
void kMeansAlgorithmSlave(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K, MPI_Datatype PointType, MPI_Datatype ClusterType, double* sumPointsCenters);
double kMeansAlgorithmMaster(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K, int limit, int numOfProccess, MPI_Datatype PointType, MPI_Datatype ClusterType, double* sumPointsCenters);
void reinitializePreviousClusterIndex(Point* points, int numOfPoints);
void recalculateClusterCenters(Cluster* clusters, int K, double* totalSumPointsCenters, int* totalClusterSize);
void gatherThePoints(Point** points, int* clustersSize, int* totalClusterSize, int K, int numOfProccess, MPI_Datatype PointType);