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

void writeToFile(double t, double q, Cluster* clusters, int K);
Point* readDataFromFile(int* totalNumOfPoints, int* K, double* limit, double* QM, double* T, double* dt);
void checkAllocation(void* pointer);
Cluster* initClusters(const Point* points, int K);
int getClosestClusterIndex(double x, double y, double z, Cluster* clusters, int K);
void groupPointsToClusters(Point** pointsMat, int* clustersSize, Point* points, int N, Cluster* clusters, int K);
double distancePointToPoint(double x1, double y1, double z1, double x2, double y2, double z2);
void calClusterCenter(Cluster* cluster, Point* clusterPoints, int clusterPointsSize);
double evaluateQuality(Point** pointsMat, Cluster* clusters, int K, int* clustersSize);
double calClusterDiameter(Point* clusterPoints, int clusterPointsSize);
void calPointsCordinates(Point* points, int totalNumOfPoints, double t);
void freeDynamicAllocation(void* pointer);
double kMeansWithIntervals(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int totalNumOfPoints, int K, double limit, double QM, double T, double dt, double* time);
double kMeansAlgorithem(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int totalNumOfPoints, int K, double limit);
