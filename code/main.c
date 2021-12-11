#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> 
#include <omp.h>

/* belong_point: cluster ao qual pertence cada ponto depois de executar o kmeans
 * numClusters: cantidade de cluster
 * filename: arquivo contendo as dimensões
 * numdim: numero de dimensões que tem cada ponto
 * num_point: cantidade de ponto
 * dim_point: contém as dimensões de cada ponto, [num_point][numdim]
 * clusters: conteém as dimensões de cada cluster [numClusters][numdim]
 */

int main( int argc, char **argv ) {
	//clock_t start, end;
	double start, end;
	
    int i, j, numClusters = 0, numdim = 0, num_point = 0, *belong_point, numthreads;    
    float **dim_point, **clusters;
    
    scanf( "%d %d %d %d", &num_point, &numdim, &numClusters, &numthreads );
    //printf( "%d %d %d\n", num_point, numClusters, numdim );
    
    dim_point = ( float** ) malloc( num_point * sizeof( float* ) );
	
	if( numClusters <= 0 ) {
		printf( "o numero de clusters deve ser maior do que 1.\n" );
		exit( -1 );
	}
    
    for( i = 0; i < num_point; i++ ){
		dim_point[ i ] = ( float* ) malloc( num_point * numdim * sizeof( float ) );
		//printf( "read num_point: %d\n", i );
		for( j = 0; j < numdim; j++ )
			scanf( "%f", &dim_point[ i ][ j ] );
	}

    clusters = ( float** ) malloc( numClusters * sizeof( float* ) );
    clusters[ 0 ] = ( float* ) malloc( numClusters * numdim * sizeof( float ) );
    
    for( i = 1; i < numClusters; i++ )
        clusters[ i ] = clusters[ i - 1 ] + numdim;

    for( i = 0; i < numClusters; i++ )
       for( j = 0; j < numdim; j++ )
          clusters[ i ][ j ] = dim_point[ i ][ j ];

    belong_point = (int*) malloc( num_point * sizeof( int ) );
    
    //start = clock(); 
    start = omp_get_wtime();

	func_kmeans( dim_point, numdim, num_point, numClusters, belong_point, clusters, numthreads );
    //func_kmeans( dim_point, numdim, num_point, numClusters, belong_point, clusters );
    
    //end = clock();
    end = omp_get_wtime();

    //printf("time: %f seg\n", ((double)( end - start ) / CLOCKS_PER_SEC ));
    printf("time omp: %lf\n", end - start);

    free( dim_point[ 0 ] );
    free( dim_point );    

    free( belong_point );
    free( clusters[ 0 ] );
    free( clusters );
    
    return 0;
}

