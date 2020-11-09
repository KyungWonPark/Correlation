/*
    -- MAGMA (version 2.5.3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date March 2020

       @author Stan Tomov
       @author Mark Gates

       @generated from testing/testing_zheevd.cpp, normal z -> c, Sun Mar 29 20:48:33 2020

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cheevd
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    /* Constants */
    const float d_zero = 0;
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;
    
    /* Local variables */
    real_Double_t   gpu_time, cpu_time;
    magmaFloatComplex *h_A, *h_R, *h_Z, *h_work, aux_work[1], unused[1];
    #ifdef COMPLEX
    float *rwork, aux_rwork[1];
    magma_int_t lrwork;
    #endif
    float *w1, *w2, result[4]={0, 0, 0, 0}, eps, abstol, runused[1];
    magma_int_t *iwork, *isuppz, *ifail, aux_iwork[1];
    magma_int_t N, Nfound, info, lwork, liwork, lda;
    eps = lapackf77_slamch( "E" );
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    // checking NoVec requires LAPACK
    opts.lapack |= (opts.check && opts.jobz == MagmaNoVec);
    
    #ifdef REAL
    if (opts.version == 3 || opts.version == 4) {
        printf("%% magma_cheevr and magma_cheevx are not available for real precisions (single, float).\n");
        return status;
    }
    #endif
    
    float tol    = opts.tolerance * lapackf77_slamch("E");
    float tolulp = opts.tolerance * lapackf77_slamch("P");
    
    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% jobz = %s, uplo = %s, ngpu = %lld\n",
           lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
           (long long) abs_ngpu);

    printf("%%   N   CPU Time (sec)   GPU Time (sec)   |S-S_magma|   |A-USU^H|   |I-U^H U|\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda = N;
            abstol = 0;  // auto, in cheevr
            
            magma_range_t range;
            magma_int_t il, iu;
            float vl, vu;
            opts.get_range( N, &range, &vl, &vu, &il, &iu );

            // query for workspace sizes
            if ( opts.version == 1 || opts.version == 2 ) {
                magma_cheevd( opts.jobz, opts.uplo,
                              N, NULL, lda, NULL,  // A, w
                              aux_work,  -1,
                              #ifdef COMPLEX
                              aux_rwork, -1,
                              #endif
                              aux_iwork, -1,
                              &info );
            }
            else if ( opts.version == 3 ) {
                #ifdef COMPLEX
                magma_cheevr( opts.jobz, range, opts.uplo,
                              N, NULL, lda,      // A
                              vl, vu, il, iu, abstol,
                              &Nfound, NULL,         // w
                              NULL, lda, NULL,   // Z, isuppz
                              aux_work,  -1,
                              #ifdef COMPLEX
                              aux_rwork, -1,
                              #endif
                              aux_iwork, -1,
                              &info );
                #endif
            }
            else if ( opts.version == 4 ) {
                #ifdef COMPLEX
                magma_cheevx( opts.jobz, range, opts.uplo,
                              N, NULL, lda,      // A
                              vl, vu, il, iu, abstol,
                              &Nfound, NULL,         // w
                              NULL, lda,         // Z
                              aux_work,  -1,
                              #ifdef COMPLEX
                              aux_rwork,
                              #endif
                              aux_iwork,
                              NULL,              // ifail
                              &info );
                // cheevx doesn't query rwork, iwork; set them for consistency
                aux_rwork[0] = float(7*N);
                aux_iwork[0] = float(5*N);
                #endif
            }
            lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
            #ifdef COMPLEX
            lrwork = (magma_int_t) aux_rwork[0];
            #endif
            liwork = aux_iwork[0];
            
            /* Allocate host memory for the matrix */
            TESTING_CHECK( magma_cmalloc_cpu( &h_A,    N*lda  ));
            TESTING_CHECK( magma_smalloc_cpu( &w1,     N      ));
            TESTING_CHECK( magma_smalloc_cpu( &w2,     N      ));
            #ifdef COMPLEX
            TESTING_CHECK( magma_smalloc_cpu( &rwork,  lrwork ));
            #endif
            TESTING_CHECK( magma_imalloc_cpu( &iwork,  liwork ));
            
            TESTING_CHECK( magma_cmalloc_pinned( &h_R,    N*lda  ));
            TESTING_CHECK( magma_cmalloc_pinned( &h_work, lwork  ));
            
            if (opts.version == 3) {
                TESTING_CHECK( magma_cmalloc_cpu( &h_Z,    N*lda      ));
                TESTING_CHECK( magma_imalloc_cpu( &isuppz, 2*max(1,N) ));
            }
            if (opts.version == 4) {
                TESTING_CHECK( magma_cmalloc_cpu( &h_Z,    N*lda      ));
                TESTING_CHECK( magma_imalloc_cpu( &ifail,  N          ));
            }
            
            /* Clear eigenvalues, for |S-S_magma| check when fraction < 1. */
            lapackf77_slaset( "Full", &N, &ione, &d_zero, &d_zero, w1, &N );
            lapackf77_slaset( "Full", &N, &ione, &d_zero, &d_zero, w2, &N );
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda );
            lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if (opts.version == 1) {
                if (opts.ngpu == 1) {
                    magma_cheevd( opts.jobz, opts.uplo,
                                  N, h_R, lda, w1,
                                  h_work, lwork,
                                  #ifdef COMPLEX
                                  rwork, lrwork,
                                  #endif
                                  iwork, liwork,
                                  &info );
                }
                else {
                    //printf( "magma_cheevd_m, ngpu %lld (%lld)\n", (long long) opts.ngpu, (long long) abs_ngpu );
                    magma_cheevd_m( abs_ngpu, opts.jobz, opts.uplo,
                                    N, h_R, lda, w1,
                                    h_work, lwork,
                                    #ifdef COMPLEX
                                    rwork, lrwork,
                                    #endif
                                    iwork, liwork,
                                    &info );
                }
            }
            else if ( opts.version == 2 ) {  // version 2: cheevdx computes selected eigenvalues/vectors
                if (opts.ngpu == 1) {
                    magma_cheevdx( opts.jobz, range, opts.uplo,
                                   N, h_R, lda,
                                   vl, vu, il, iu,
                                   &Nfound, w1,
                                   h_work, lwork,
                                   #ifdef COMPLEX
                                   rwork, lrwork,
                                   #endif
                                   iwork, liwork,
                                   &info );
                }
                else {
                    //printf( "magma_cheevdx_m, ngpu %lld (%lld)\n", (long long) opts.ngpu, (long long) abs_ngpu );
                    magma_cheevdx_m( abs_ngpu, opts.jobz, range, opts.uplo,
                                     N, h_R, lda,
                                     vl, vu, il, iu,
                                     &Nfound, w1,
                                     h_work, lwork,
                                     #ifdef COMPLEX
                                     rwork, lrwork,
                                     #endif
                                     iwork, liwork,
                                     &info );
                }
                //printf( "il %lld, iu %lld, Nfound %lld\n", (long long) il, (long long) iu, (long long) Nfound );
            }
            else if ( opts.version == 3 ) {  // version 3: MRRR, computes selected eigenvalues/vectors
                // only complex version available
                #ifdef COMPLEX
                magma_cheevr( opts.jobz, range, opts.uplo,
                              N, h_R, lda,
                              vl, vu, il, iu, abstol,
                              &Nfound, w1,
                              h_Z, lda, isuppz,
                              h_work, lwork,
                              #ifdef COMPLEX
                              rwork, lrwork,
                              #endif
                              iwork, liwork,
                              &info );
                lapackf77_clacpy( "Full", &N, &N, h_Z, &lda, h_R, &lda );
                #endif
            }
            else if ( opts.version == 4 ) {  // version 3: cheevx (QR iteration), computes selected eigenvalues/vectors
                // only complex version available
                #ifdef COMPLEX
                magma_cheevx( opts.jobz, range, opts.uplo,
                              N, h_R, lda,
                              vl, vu, il, iu, abstol,
                              &Nfound, w1,
                              h_Z, lda,
                              h_work, lwork,
                              #ifdef COMPLEX
                              rwork, /*lrwork,*/
                              #endif
                              iwork, /*liwork,*/
                              ifail,
                              &info );
                lapackf77_clacpy( "Full", &N, &N, h_Z, &lda, h_R, &lda );
                #endif
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_cheevd returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            bool okay = true;
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvst routine.
                   A is factored as A = U S U^H and the following 3 tests computed:
                   (1)    | A - U S U^H | / ( |A| N )
                   (2)    | I - U^H U   | / ( N )
                   (3)    | S(with U) - S(w/o U) | / | S |    // currently disabled, but compares to LAPACK
                   =================================================================== */
                magmaFloatComplex *work;
                TESTING_CHECK( magma_cmalloc_cpu( &work, 2*N*N ));
                
                // e is unused since kband=0; tau is unused since itype=1
                lapackf77_chet21( &ione, lapack_uplo_const(opts.uplo), &N, &izero,
                                  h_A, &lda,
                                  w1, runused,
                                  h_R, &lda,
                                  h_R, &lda,
                                  unused, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[0] );
                result[0] *= eps;
                result[1] *= eps;
                
                magma_free_cpu( work );  work=NULL;
                
                // Disable third eigenvalue check that calls routine again --
                // it obscures whether error occurs in first call above or in this call.
                // But see comparison to LAPACK below.
                //
                //lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
                //magma_cheevd( MagmaNoVec, opts.uplo,
                //              N, h_R, lda, w2,
                //              h_work, lwork,
                //              #ifdef COMPLEX
                //              rwork, lrwork,
                //              #endif
                //              iwork, liwork,
                //              &info );
                //if (info != 0) {
                //    printf("magma_cheevd returned error %lld: %s.\n",
                //           (long long) info, magma_strerror( info ));
                //}
                //
                //float maxw=0, diff=0;
                //for( int j=0; j < N; j++ ) {
                //    maxw = max(maxw, fabs(w1[j]));
                //    maxw = max(maxw, fabs(w2[j]));
                //    diff = max(diff, fabs(w1[j]-w2[j]));
                //}
                //result[2] = diff / (N*maxw);
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                if ( opts.version == 1 || opts.version == 2 ) {
                    lapackf77_cheevd( lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda, w2,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork, &lrwork,
                                      #endif
                                      iwork, &liwork,
                                      &info );
                }
                else if ( opts.version == 3 ) {
                    lapackf77_cheevr( lapack_vec_const(opts.jobz),
                                      lapack_range_const(range),
                                      lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda,
                                      &vl, &vu, &il, &iu, &abstol,
                                      &Nfound, w2,
                                      h_Z, &lda, isuppz,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork, &lrwork,
                                      #endif
                                      iwork, &liwork,
                                      &info );
                    lapackf77_clacpy( "Full", &N, &N, h_Z, &lda, h_A, &lda );
                }
                else if ( opts.version == 4 ) {
                    lapackf77_cheevx( lapack_vec_const(opts.jobz),
                                      lapack_range_const(range),
                                      lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda,
                                      &vl, &vu, &il, &iu, &abstol,
                                      &Nfound, w2,
                                      h_Z, &lda,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork,
                                      #endif
                                      iwork,
                                      ifail,
                                      &info );
                    lapackf77_clacpy( "Full", &N, &N, h_Z, &lda, h_A, &lda );
                }
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_cheevd returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                // compare eigenvalues
                float maxw=0, diff=0;
                for( int j=0; j < N; j++ ) {
                    maxw = max(maxw, fabs(w1[j]));
                    maxw = max(maxw, fabs(w2[j]));
                    diff = max(diff, fabs(w1[j] - w2[j]));
                }
                result[3] = diff / (N*maxw);
                
                okay = okay && (result[3] < tolulp);
                printf("%5lld   %9.4f        %9.4f         %8.2e  ",
                       (long long) N, cpu_time, gpu_time, result[3] );
            }
            else {
                printf("%5lld      ---           %9.4f           ---     ",
                       (long long) N, gpu_time);
            }
            
            // print error checks
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                okay = okay && (result[0] < tol) && (result[1] < tol);
                printf("    %8.2e    %8.2e", result[0], result[1] );
            }
            else {
                printf("      ---         ---   ");
            }
            printf("   %s\n", (okay ? "ok" : "failed"));
            status += ! okay;
            
            magma_free_cpu( h_A   );
            magma_free_cpu( w1    );
            magma_free_cpu( w2    );
            #ifdef COMPLEX
            magma_free_cpu( rwork );
            #endif
            magma_free_cpu( iwork );
            
            magma_free_pinned( h_R    );
            magma_free_pinned( h_work );
            
            if ( opts.version == 3 ) {
                magma_free_cpu( h_Z    );
                magma_free_cpu( isuppz );
            }
            if ( opts.version == 4 ) {
                magma_free_cpu( h_Z    );
                magma_free_cpu( ifail  );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}