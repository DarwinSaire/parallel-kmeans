#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h> 
#include <fstream>
#include <iostream>
#include <omp.h>
#include "kmeans.c"

using namespace std;

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
    double times[ 5 ] = { 0.0 };
    string inputs[ 4 ] = { "input/dim128.txt", "input/dim256.txt", "input/dim512.txt", "input/dim1024.txt" };
    int number_threads[ 5 ] =  { 1, 2, 4, 8, 16 };
    ofstream myfile_out ("output.txt");

    for( int n_input = 0; n_input < 4; n_input++ ){

        ifstream myfile_in( inputs[ n_input ].c_str() );

	cout << "---------------- " << inputs[ n_input ] << " ----------------\n";

        int i, j, numClusters = 0, numdim = 0, num_point = 0, *belong_point, numthreads;    
        float **dim_point, **clusters;

        myfile_in >> num_point >> numdim >> numClusters >> numthreads;
        //scanf( "%d %d %d %d", &num_point, &numdim, &numClusters, &numthreads );
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
                myfile_in >> dim_point[ i ][ j ];
            //scanf( "%f", &dim_point[ i ][ j ] );
        }

	myfile_in.close();

        for( int n_threads = 0; n_threads < 5; n_threads++ ){
	    
            numthreads = number_threads[ n_threads ];

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
            printf( "n_threads: %d\ttime: %.4lf", numthreads, end - start );
	    times[ n_threads ] = end - start;
	    if( n_threads > 0 ) printf( "\tspeedup: %.4lf", times[ 0 ] / times[ n_threads ] );
	    printf( "\n" );

	for( int k = 0; k < num_point; k++ ) 
		myfile_out << belong_point[ k ] << " ";
 	myfile_out << endl << endl;

            free( belong_point );
            free( clusters[ 0 ] );
            free( clusters );
        }
        free( dim_point[ 0 ] );
        free( dim_point );    
    }
    myfile_out.close();
    return 0;
}

