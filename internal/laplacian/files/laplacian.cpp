#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "include/magma_v2.h"
#include "include/magma_lapack.h"
#include "include/magma_operators.h"

// Macros
#define TESTING_CHECK( err ) 												 \
	do { 																	 \
		magma_int_t err_ = ( err ); 										 \
		if ( err_ != 0 ) { 													 \
			fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
					 #err, __FILE__, __LINE__, 								 \
					 (long long) err_, magma_strerror(err_) ); 				 \
			exit(1); 														 \
		} 																	 \
	} while( 0 ) 															 \

// 
int eigenDecomposition(int jobSize, double *pMatBuffer, double *pEigVal) {

	// MAGMA Init
	TESTING_CHECK( magma_init() );

	// DSYEVD routine parameters 
	magma_int_t ngpu = 4;
	magma_vec_t jobz = MagmaVec;
	magma_uplo_t uplo = MagmaUpper;
	magma_int_t N = (magma_int_t) jobSize;
	double* h_R;
	magma_int_t lda = N;
	double* w1;
	double* h_work;
	magma_int_t lwork;
	magma_int_t* iwork;
	magma_int_t liwork;
	magma_int_t info;

	// Query parameters
	double aux_work[1];
	magma_int_t aux_iwork[1];

	// Query for workspace sizes
	magma_dsyevd(jobz, uplo, N, NULL, lda, NULL, aux_work, -1, aux_iwork, -1, &info);
	
	lwork = (magma_int_t) aux_work[0];
	liwork = aux_iwork[0];

	// Allocate host memory for workload
	TESTING_CHECK( magma_dmalloc_pinned( &h_R, N*lda ));
	TESTING_CHECK( magma_dmalloc_cpu( &w1, N));
	TESTING_CHECK( magma_dmalloc_pinned( &h_work, lwork ));
	TESTING_CHECK( magma_imalloc_cpu( &iwork, liwork ));

	// Copy matrix into MAGMA pinned memory
	lapackf77_dlacpy( MagmaFullStr, &N, &N, pMatBuffer, &lda, h_R, &lda);

	// Carry out calculations
	magma_dsyevd_m(ngpu, jobz, uplo, N, h_R, lda, w1, h_work, lwork, iwork, liwork, &info);
	if (info != 0) {
		printf("lapackf77_dsyevd returned error: %s.\n", magma_strerror( info ));
		return -1;
	}

	// Copy result back
	lapackf77_dlacpy( MagmaFullStr, &N, &N, h_R, &lda, pMatBuffer, &lda);
	for (int i = 0; i < N; i++) {
		pEigVal[i] = w1[i];
	}

	// Finish
	magma_free_cpu( w1 );
	magma_free_cpu( iwork );
	magma_free_pinned( h_R );
	magma_free_pinned( h_work );

	TESTING_CHECK( magma_finalize() );
	return 0;
}

#ifdef __cplusplus
}
#endif