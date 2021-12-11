#include <stdio.h>
#include <stdlib.h>
//#include "kmeans.h"


/* func_distance: obtem a distância euclidiana de dois pontos multidimensional 
 * numdim: número de dimensões
 * dim_pointX: dimensões de ponto X
 * dim_pointY: dimensões de ponto Y*
 */
float func_distance(int numdim, float *dim_pointX, float *dim_pointY ){
    int i;
    float ans = 0.0;
    for( i = 0; i < numdim; i++ )
        ans += ( dim_pointX[ i ] - dim_pointY[ i ] ) * (dim_pointX[ i ]- dim_pointY[ i ] );
    return(ans);
}

/* func_find_nearest_cluster: obtem o cluster mais proximo de um determinado ponto e retornar o id do cluster.
 * numcluster: número de clusters
 * numdim: número de dimensões do ponto
 * dim_point: dimensões de ponto
 * clusters: dimensões de clusters[numclusters][numdim]
 */
int func_find_nearest_cluster(int numClusters, int numdim, float  *dim_point, float **clusters){
    int i, nearest_cluster = 0;
    float dist;
    float min_dist = func_distance( numdim, dim_point, clusters[ 0 ] );

    for( i = 1; i < numClusters; i++ ){
        dist = func_distance( numdim, dim_point, clusters[ i ] );
        if( dist < min_dist ){
            min_dist = dist;
            nearest_cluster    = i;
        }
    }
    return nearest_cluster;
}

/*func_kmeans: obtem o melhor agrupamento, com a melhor posição dos clusters, retornando belong_point e clusters
 * dim_point: contém as dimensões de cada ponto, [num_point][numdim]
 * numdim: numero de dimensões que tem cada ponto
 * num_point: cantidade de ponto
 * numClusters: cantidade de cluster
 * clusters: conteém as dimensões de cada cluster [numClusters][numdim]
 * belong_point: cluster ao qual pertence cada ponto depois de executar o kmeans
 */

int func_kmeans(float **dim_point, int numdim, int num_point, int numClusters, int *belong_point, float **clusters, int numthreads ){
    int i, j, nearest_cluster, counter = 0;
    /*numero de objetos que tem cada cluster*/
    int *newClusterSize;
    /*dimensões da nova mudanca dos clusters*/
    float **newClusters;
    
    newClusterSize = ( int* )calloc( numClusters, sizeof( int ) );
    newClusters = ( float** )malloc( numClusters * sizeof( float* ) );
    newClusters[ 0 ] = ( float* )calloc( numClusters * numdim, sizeof( float ) );

    for( i = 0; i < num_point; i++ ) belong_point[ i ] = -1;

    for( i = 1; i < numClusters; i++ )
        newClusters[ i ] = newClusters[ i - 1 ] + numdim;

    do {
		/*selecionar o cluster mais próximo para cada ponto e sumar as dimensões em newCluster
		  para obtem um novo cluster*/
		#pragma omp parallel for num_threads(numthreads) \
        private(i,j,nearest_cluster) \
        firstprivate(num_point,numClusters,numdim) \
        shared(dim_point,clusters,belong_point,newClusters,newClusterSize) \
        schedule(static) 
        for( i = 0; i < num_point; i++ ){
            nearest_cluster = func_find_nearest_cluster( numClusters, numdim, dim_point[ i ], clusters);
            belong_point[ i ] = nearest_cluster;
	   #pragma omp atomic
            newClusterSize[ nearest_cluster ]++;
            for( j = 0; j < numdim; j++ )
		#pragma omp atomic
                newClusters[ nearest_cluster ][ j ] += dim_point[ i ][ j ];
        }

		/*obter novos clusters( medía ) e substituir os clusters antigos para os novos clusters*/
        for( i = 0; i < numClusters; i++ ){
            for( j = 0; j < numdim; j++ ){
                if( newClusterSize[ i ] > 0 )
                    clusters[ i ][ j ] = newClusters[ i ][ j ] / newClusterSize[ i ];
                newClusters[ i ][ j ] = 0.0;
            }
            newClusterSize[ i ] = 0;
        }
    } while( counter++ < 1000 );

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return 1;
}

