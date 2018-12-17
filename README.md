
# KMeansParallelAlgorithm

## Problem Definition

Given a set of points in 3-dimensional space. Initial position (xi, yi, zi) and velocity (vxi, vyi, vzi) are known for each point Pi. Its position at the given time t can be calculated as follows:

xi(t) = xi + t*vxi

yi(t) = yi + t*vyi

zi(t) = zi + t*vzi

Implement simplified K-Means algorithm to find K clusters. Find a first occurrence during given time interval [0, T] when a system of K clusters has a Quality Measure q that is less than given value QM.


## Simplified K-Means algorithm

1.	Choose first K points as centers of clusters.

2.	Group points around the given cluster centers - for each point define a center that is most close to the point.

3.	Recalculate the cluster centers (average of all points in the cluster)

4.	Check the termination condition – no points move to other clusters or maximum iteration LIMIT was made.

5.	Repeat from 2 till the termination condition fulfills.

6.	Evaluate the Quality of the clusters found. Calculate diameter of each cluster – maximum distance between any two points in this cluster. The Quality is equal to an average of ratio diameters of the cluster divided by distance to other clusters. For example, in case of k = 3 the quality is equal 
q = (d1/D12 + d1/D13 + d2/D21 + d2/D23 + d3/D31 + d3/D32) / 6, 
where di is a diameter of cluster i and Dij is a distance between centers of cluster i and cluster j.



## Input data and Output Result of the project

You will be supplied with the following data 

•	N - number of points

•	K - number of clusters to find

•	LIMIT – the maximum number of iterations for K-MEAN algorithm. 

•	QM – quality measure to stop

•	T – defines the end of time interval [0, T]

•	dT – defines moments t = n*dT, n = { 0, 1, 2, … , T/dT} for which calculate the clusters and the quality

•	Coordinates and Velocities of all points

## Implementation:

1.	The master read and obtain all the points from the input file.

2.	 The master process does initial to the clusters according to the first k points.

3.	The master process calculates the amount of points that each process (include him) will handle with, also, the master handle with the rest of the points (in case N%numproc!=0).

4.	 The master process broadcast to the other processes the number of clusters and number of points that the individual process will be handle.

5.	The master process sends to each slave process that appropriate segment of points.

6.	The all processes together (master and slaves) start the Algorithm.

7.	The master process sends to each slave process the current time (OMP).

8.	Each process calculates his own points coordinates (CUDA) according to the time from step 7. 

9.	 Each process activate the K-Means function.

    K-Means Algorithm:
    •	The master process sends the calculated clusters to the slave’s processes(OMP).
    
    •	Each process iterates on all his own points and finds the closest cluster to each of the points (OMP).
    
    •	Each process computes the sum of coordinates x, sum of coordinates y and sum of coordinates z of all the points who belong to the same cluster. Each cluster have his own sum X, sum Y and sum Z.(OMP)
    
    •	With the help of the MPI_Reduce, all the sums of x, y and z are gathering in the master process. 
    
    Also, the total number of points who are belong to the same cluster are gathering together in the Master process (MPI_SUM).
    
    •	The Master process, calculated and updated the clusters centers according to the sum x coordinates, sum y coordinates, sum z coordinates and the number of points who belong to specific cluster(OMP).
    
    •	Each process checks if his all of points belongs to the same cluster in step 2 of the K-Means algorithm.
    
    •	Each process sends the answer from the previous step to the master process.
    
    •	The master process gathers the answers from the step 6 with the help of the MPI_Reduce and checks if the termination condition is fulfilled. 
    
    •	If the termination condition is fulfilling or the master process has done limit iterations: the master process gathers all the points from all the slave processes, and send K-Means-Termination tag to the slaves and finally calculate and return the quality.
    
    •	If the termination condition is not fulfilled and the master process has not done limit iteration:  return to step 1 in the K-Means algorithm. 

10.	The master obtains the current quality and checks if the quality is less than QM and check:

    •	If the current quality is less than QM or the time is T/dt the master sends to all slave processes Final-Termination-tag and return the quality.

    •	If the current quality is greater than QM, the master saves the less quality between the current quality and the previous quality.

11.	The slaves’ processes are finished and finalize.

12.	The master is writing to the output file the time, quality and the clusters centers to a output file and finalize.
