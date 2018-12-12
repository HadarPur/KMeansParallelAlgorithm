
# KMeansParallelAlgorithm

## Problem Definition
Given a set of points in 3-dimensional space. Initial position (xi, yi, zi) and velocity (vxi, vyi, vzi) are known for each point Pi. Its position at the given time t can be calculated as follows:\n
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

