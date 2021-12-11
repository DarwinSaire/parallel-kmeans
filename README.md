# Parallel K-means
Parallelization of K-Means Clustering Algorithm. OpenMP and CUDA

Dataset for clustering, http://cs.joensuu.fi/sipu/datasets

Input

- total number of points
- number of dimensions for each point
- number of clusters (K)
- number of threads used
- feature vector for each point

Output

- An array with the cluster to which each point belongs

  

Parallel k-mean algorithm

<img src="img/parallel-kmeans.png" alt="parallel-kmean" width="700">



Speedup table

<img src="img/speedup-kmeans.png" alt="speed-kmean" width="600">



#### Run the code

General C

```c
gcc -g main.c kmeans.c -o main -fopenmp -lm
```



General C++

```c++
g++ -g main_general.cpp -o main_general -fopenmp -lm
```



CUDA

```nvcc
nvcc -arch=sm_20 main_final.cu -o main_final
```

