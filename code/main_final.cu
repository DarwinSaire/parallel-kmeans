#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> 
#include <sys/types.h>
#include <math.h>
#include <errno.h>

__host__ __device__ float func_distance(int numdim, float *dim_pointX, int id_point, float *dim_pointY, int id_cluster ){
    int i;
    float ans = 0.0;
    for( i = 0; i < numdim; i++ )
        ans += ( dim_pointX[ id_point * numdim + i ] - dim_pointY[ id_cluster * numdim + i ] ) * ( dim_pointX[ id_point * numdim + i ]- dim_pointY[ id_cluster * numdim + i ] );
    return ans;
}

__host__ __device__ int func_find_nearest_cluster(int numClusters, int numdim, float  *dim_point, float *clusters, int id_point ){
    int i, nearest_cluster = 0;
    float dist;
    float min_dist = func_distance( numdim, dim_point, id_point, clusters, 0 );

    for( i = 1; i < numClusters; i++ ){
        dist = func_distance( numdim, dim_point, id_point, clusters, i );
        if( dist < min_dist ){
            min_dist = dist;
            nearest_cluster  = i;
        }
    }
    return nearest_cluster;
}

__global__ void Func_kmeans_point(float *dim_point, int numdim, int num_point, int numClusters, int *belong_point, float *clusters, float *newClusters, int *newClusterSize ){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i > num_point ) return;

    int nearest_cluster;
    if( i < num_point ) {
	nearest_cluster = func_find_nearest_cluster( numClusters, numdim, dim_point, clusters, i );
	belong_point[ i ] = nearest_cluster;
	atomicAdd( &newClusterSize[ nearest_cluster ], 1 );
	//newClusterSize[ nearest_cluster ] += 1;
	for( int j = 0; j < numdim; j++ ){
		float num = dim_point[ i * numdim + j ];
		atomicAdd( &newClusters[ nearest_cluster * numdim + j ], num );
	}
		//newClusters[ nearest_cluster * numdim + j ] += dim_point[ i * numdim + j ] ;
    } 
}

int func_kmeans(float *dim_point, int numdim, int num_point, int numClusters, int *belong_point, float *clusters, int numthreads ){
    int i, j, counter = 0;
    /*numero de objetos que tem cada cluster*/
    int *newClusterSize;
    /*dimensões da nova mudanca dos clusters*/
    float *newClusters;

    //dim_point, clusters, newClusters
    
    newClusterSize = ( int* )calloc( numClusters, sizeof( int ) );
    newClusters = ( float* )calloc( numClusters * numdim, sizeof( float ) );

    for( i = 0; i < num_point; i++ ) belong_point[ i ] = -1;

    /*for( i = 1; i < numClusters; i++ )
        newClusters[ i ] = newClusters[ i - 1 ] + numdim;*/

    float * device_dim_point, *device_clusters, *device_newclusters;
    int *device_belong_point, *device_newclustersize;

    int size_device_dimpoint = num_point * numdim * sizeof( float );
    int size_device_clusters = numClusters * numdim * sizeof( float );
    int size_device_newclusters = numClusters * numdim * sizeof( float );
    int size_device_belong = num_point * sizeof( int );
    int size_device_newclusersize = numClusters * sizeof( int );

    cudaMalloc( ( void** ) &device_dim_point, size_device_dimpoint );
    cudaMalloc( ( void** ) &device_clusters, size_device_clusters );
    cudaMalloc( ( void** ) &device_newclusters, size_device_newclusters );
    cudaMalloc( ( void** ) &device_belong_point, size_device_belong );
    cudaMalloc( ( void** ) &device_newclustersize, size_device_newclusersize );

    cudaMemset(device_newclusters,0,size_device_newclusters);

    cudaMemcpy( device_dim_point, dim_point, size_device_dimpoint, cudaMemcpyHostToDevice );

    int threads_block = 256;
    dim3 DimGrid( ( num_point - 1 ) / threads_block + 1, 1, 1 );
    dim3 DimBlock( threads_block, 1, 1 );

    do {
	/*selecionar o cluster mais próximo para cada ponto e sumar as dimensões em newCluster
	  para obtem um novo cluster*/

	cudaMemcpy( device_clusters, clusters, size_device_clusters, cudaMemcpyHostToDevice );
	cudaMemcpy( device_newclusters, newClusters, size_device_newclusters, cudaMemcpyHostToDevice );
	cudaMemcpy( device_belong_point, belong_point, size_device_belong, cudaMemcpyHostToDevice );
	cudaMemcpy( device_newclustersize, newClusterSize, size_device_newclusersize, cudaMemcpyHostToDevice );

	//(float * dim_point, int numdim, int num_point, int numClusters, int *belong_point, float *clusters, int *newClusterSize, int *newClusters )

	Func_kmeans_point<<< DimGrid, DimBlock >>>( device_dim_point, numdim, num_point, numClusters, device_belong_point, device_clusters, device_newclusters, device_newclustersize );

	cudaMemcpy( clusters, device_clusters, size_device_clusters, cudaMemcpyDeviceToHost );
	cudaMemcpy( newClusters ,device_newclusters, size_device_newclusters, cudaMemcpyDeviceToHost );
	cudaMemcpy( belong_point, device_belong_point, size_device_belong, cudaMemcpyDeviceToHost );
	cudaMemcpy( newClusterSize, device_newclustersize, size_device_newclusersize, cudaMemcpyDeviceToHost );

        /*for( i = 0; i < num_point; i++ ){
            nearest_cluster = func_find_nearest_cluster( numClusters, numdim, dim_point[ i ], clusters);
            belong_point[ i ] = nearest_cluster;
            newClusterSize[ nearest_cluster ]++;
            for( j = 0; j < numdim; j++ )
                newClusters[ nearest_cluster ][ j ] += dim_point[ i ][ j ];
        }*/

	/*obter novos clusters( medía ) e substituir os clusters antigos para os novos clusters*/
        for( i = 0; i < numClusters; i++ ){
            for( j = 0; j < numdim; j++ ){
                if( newClusterSize[ i ] > 0 )
                    clusters[ i * numdim + j ] = newClusters[ i  * numdim + j ] / newClusterSize[ i ];
                newClusters[ i * numdim + j ] = 0.0;
            }
            newClusterSize[ i ] = 0;
        }
    } while( counter++ < 1000 );

    //for( i = 0; i < num_point; i++ )
	//printf( "%d %d\n", i, belong_point[ i ] );

    //free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return 1;
}

/* belong_point: cluster ao qual pertence cada ponto depois de executar o kmeans
 * numClusters: cantidade de cluster
 * filename: arquivo contendo as dimensões
 * numdim: numero de dimensões que tem cada ponto
 * num_point: cantidade de ponto
 * dim_point: contém as dimensões de cada ponto, [num_point][numdim]
 * clusters: conteém as dimensões de cada cluster [numClusters][numdim]
 */

int main( int argc, char **argv ) {
	clock_t start, end;
	
    int i, j, numClusters = 0, numdim = 0, num_point = 0, *belong_point, numthreads;    
    float *dim_point, *clusters; //**dim_point, **clusters;
    //double timing, io_timing, clustering_timing;
    
    scanf( "%d %d %d %d", &num_point, &numdim, &numClusters, &numthreads );
    //printf( "%d %d %d\n", num_point, numClusters, numdim );
    
    dim_point = ( float* ) malloc( num_point * numdim * sizeof( float ) );
	
    if( numClusters <= 0 ) {
	printf( "o numero de clusters deve ser maior do que 1.\n" );
	exit( -1 );
    }
    
    for( i = 0; i < num_point; i++ ){
		//printf( "read num_point: %d\n", i );
		for( j = 0; j < numdim; j++ )
			scanf( "%f", &dim_point[ i * numdim + j ] );
	}

    clusters = ( float* ) malloc( numClusters * numdim * sizeof( float ) );
    
    /*for( i = 1; i < numClusters; i++ )
        clusters[ i ] = clusters[ i - 1 ] + numdim;*/

    for( i = 0; i < numClusters; i++ )
       for( j = 0; j < numdim; j++ )
          clusters[ i * numdim + j ] = dim_point[ i * numdim + j ];

    belong_point = (int*) malloc( num_point * sizeof( int ) );
    
    start = clock(); 

	func_kmeans( dim_point, numdim, num_point, numClusters, belong_point, clusters, numthreads );
    //func_kmeans( dim_point, numdim, num_point, numClusters, belong_point, clusters );
    
    end = clock();

    printf("time: %f seg\n", ((double)( end - start ) / CLOCKS_PER_SEC ));

    //free( dim_point[ 0 ] );
    free( dim_point );    

    free( belong_point );
    //free( clusters[ 0 ] );
    free( clusters );
    
    return 0;
}

