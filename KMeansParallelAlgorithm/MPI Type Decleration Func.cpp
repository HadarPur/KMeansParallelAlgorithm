#include "KMeans Header.h"
#include "MPI Type Decleration Header.h"

// Define Point MPI Type 
void createPointType(MPI_Datatype* PointType)
{
	Point point;
	MPI_Datatype PointMPIType[POINT_STRUCT_SIZE] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
		MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
		MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
		MPI_INT, MPI_INT };

	int pointBlockLen[POINT_STRUCT_SIZE] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint pointDisp[POINT_STRUCT_SIZE];
	pointDisp[0] = (char*)&point.x0 - (char*)&point;
	pointDisp[1] = (char*)&point.y0 - (char*)&point;
	pointDisp[2] = (char*)&point.z0 - (char*)&point;
	pointDisp[3] = (char*)&point.x - (char*)&point;
	pointDisp[4] = (char*)&point.y - (char*)&point;
	pointDisp[5] = (char*)&point.z - (char*)&point;
	pointDisp[6] = (char*)&point.vx - (char*)&point;
	pointDisp[7] = (char*)&point.vy - (char*)&point;
	pointDisp[8] = (char*)&point.vz - (char*)&point;
	pointDisp[9] = (char*)&point.currentClusterIndex - (char*)&point;
	pointDisp[10] = (char*)&point.previousClusterIndex - (char*)&point;

	MPI_Type_create_struct(POINT_STRUCT_SIZE, pointBlockLen, pointDisp, PointMPIType, PointType);
	MPI_Type_commit(PointType);
}

// Define Cluster MPI Type 
void createClusterType(MPI_Datatype* ClusterType)
{
	Cluster cluster;
	MPI_Datatype ClusterMPIType[CLUSTER_STRUCT_SIZE] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

	int clusterBlockLen[CLUSTER_STRUCT_SIZE] = { 1, 1, 1, 1 };
	MPI_Aint clusterDisp[CLUSTER_STRUCT_SIZE];
	clusterDisp[0] = (char*)&cluster.x - (char*)&cluster;
	clusterDisp[1] = (char*)&cluster.y - (char*)&cluster;
	clusterDisp[2] = (char*)&cluster.z - (char*)&cluster;
	clusterDisp[3] = (char*)&cluster.diameter - (char*)&cluster;

	MPI_Type_create_struct(CLUSTER_STRUCT_SIZE, clusterBlockLen, clusterDisp, ClusterMPIType, ClusterType);
	MPI_Type_commit(ClusterType);
}