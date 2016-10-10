/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "irregular_tiled_matrix.h"
#include "summa_z.h"

#define FMULS_SUMMA(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))
#define FADDS_SUMMA(__m, __n, __k) ((double)(__m) * (double)(__n) * (double)(__k))
#define FLOPS_ZSUMMA(__m, __n, __k) (6. * FMULS_SUMMA((__m), (__n), (__k)) + 2.0 * FADDS_SUMMA((__m), (__n), (__k)) )


static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum transA, PLASMA_enum transB,
                           dague_complex64_t alpha, int Am, int An, int Aseed,
                           int Bm, int Bn, int Bseed,
                           int M,  int N,
                           irregular_tiled_matrix_desc_t *ddescCfinal );



//static unsigned long long int Rnd64seed = 100;
#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

static void* init_tile(int mb, int nb, unsigned long long int *seed)
{
    unsigned long long int ran = *seed;
    int i, j;
    dague_complex64_t *array = (dague_complex64_t*)malloc(sizeof(dague_complex64_t)*mb*nb);

    for (i = 0; i < mb; ++i)
        for (j = 0; j < nb; ++j) {
            array[i*mb+j] = ran;
            ran = Rnd64_A * ran + Rnd64_C;
        }
    *seed = ran;
    return array;
}

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    unsigned long long int Aseed = 3872;
    unsigned long long int Bseed = 4674;
    int tA = PlasmaNoTrans;
    int tB = PlasmaNoTrans;
    dague_complex64_t alpha =  0.51;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(DAGUE_HAVE_CUDA) && 1
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int cores = iparam[IPARAM_NCORES];
    int gpus  = iparam[IPARAM_NGPUS];
    int P     = iparam[IPARAM_P];
    int Q     = iparam[IPARAM_Q];
    int M     = iparam[IPARAM_M];
    int N     = iparam[IPARAM_N];
    int K     = iparam[IPARAM_K];
    int NRHS  = K;
    int LDA   = max(M, iparam[IPARAM_LDA]);
    int LDB   = max(N, iparam[IPARAM_LDB]);
    int LDC   = max(K, iparam[IPARAM_LDC]);
    int IB    = iparam[IPARAM_IB];
    int MB    = iparam[IPARAM_MB];
    int NB    = iparam[IPARAM_NB];
    int SMB   = iparam[IPARAM_SMB];
    int SNB   = iparam[IPARAM_SNB];
    int HMB   = iparam[IPARAM_HMB];
    int HNB   = iparam[IPARAM_HNB];
    int MT    = (M%MB==0) ? (M/MB) : (M/MB+1);
    int NT    = (N%NB==0) ? (N/NB) : (N/NB+1);
    int KT    = (K%MB==0) ? (K/MB) : (K/MB+1);
    int check = iparam[IPARAM_CHECK];
    int check_inv = iparam[IPARAM_CHECKINV];
    int loud  = iparam[IPARAM_VERBOSE];
    int scheduler = iparam[IPARAM_SCHEDULER];
    int random_seed = iparam[IPARAM_RANDOM_SEED];
    int matrix_init = iparam[IPARAM_MATRIX_INIT];
    int butterfly_level = iparam[IPARAM_BUT_LEVEL];
    int async = iparam[IPARAM_ASYNC];
    (void)cores;(void)gpus;(void)P;(void)Q;(void)M;(void)N;(void)K;(void)NRHS; \
    (void)LDA;(void)LDB;(void)LDC;(void)IB;(void)MB;(void)NB;(void)MT;(void)NT;(void)KT; \
    (void)SMB;(void)SNB;(void)HMB;(void)HNB;(void)check;(void)loud;(void)async; \
    (void)scheduler;(void)butterfly_level;(void)check_inv;(void)random_seed;(void)matrix_init;

    PASTE_CODE_FLOPS(FLOPS_ZSUMMA, ((DagDouble_t)M,(DagDouble_t)N,(DagDouble_t)K));

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    unsigned int *Mtiling = (unsigned int*)malloc(MT*sizeof(unsigned int));
    unsigned int *Ktiling = (unsigned int*)malloc(KT*sizeof(unsigned int));
    unsigned int *Ntiling = (unsigned int*)malloc(KT*sizeof(unsigned int));

    int i, j, k, l;
    for (i = 0; i < MT; ++i) Mtiling[i] = MB;
    if (M%MB != 0) Mtiling[MT-1] = M%MB;
    int KB = 1+(K-1)/KT;
    for (i = 0; i < KT; ++i) Ktiling[i] = KB;
    if (K%KB != 0) Ktiling[KT-1] = K%KB;
    for (i = 0; i < NT; ++i) Ntiling[i] = NB;
    if (N%NB != 0) Ntiling[NT-1] = N%NB;


    fprintf(stdout, "(MT = %d, MB = %d) x (KT = %d, KB = %d) x (NT = %d, NB = %d)\n", MT, MB, KT, KB, NT, NB);
    fprintf(stdout, "M tiling:");
    for (i = 0; i < MT; ++i) fprintf(stdout, " %d", Mtiling[i]);
    fprintf(stdout, "\n");
    fprintf(stdout, "K tiling:");
    for (i = 0; i < KT; ++i) fprintf(stdout, " %d", Ktiling[i]);
    fprintf(stdout, "\n");
    fprintf(stdout, "N tiling:");
    for (i = 0; i < NT; ++i) fprintf(stdout, " %d", Ntiling[i]);
    fprintf(stdout, "\n");

    irregular_tiled_matrix_desc_t ddescC;
    irregular_tiled_matrix_desc_init(&ddescC, tile_coll_ComplexDouble,
                                     nodes, rank, M, N, MT, NT,
                                     Mtiling, Ntiling,
                                     0, 0, 0, 0, P);

    fprintf(stdout, "check = %d\n", check);
    /* initializing matrix structure */
    if(!check) {
	    irregular_tiled_matrix_desc_t ddescA;
	    irregular_tiled_matrix_desc_init(&ddescA, tile_coll_ComplexDouble,
                                         nodes, rank, M, K, MT, KT,
	                                     Mtiling, Ktiling,
                                         0, 0, 0, 0, P);

        irregular_tiled_matrix_desc_t ddescB;
        irregular_tiled_matrix_desc_init(&ddescB, tile_coll_ComplexDouble,
                                         nodes, rank, K, N, KT, NT,
                                         Ktiling, Ntiling,
                                         0, 0, 0, 0, P);

        /* matrix generation */
        void *ptr;
        if(1 || loud > 2) printf("+++ Generate matrices ... ");
        for (i = ddescA.grid.rrank*ddescA.grid.strows; i < MT; i+=ddescA.grid.rows*ddescA.grid.strows)
	        for (k = 0; k < ddescA.grid.stcols; ++k)
		        for (j = ddescA.grid.crank*ddescA.grid.stcols; j < KT; j+=ddescA.grid.cols*ddescA.grid.stcols)
			        for (l = 0; l < ddescA.grid.stcols; ++l) {
				        ptr = init_tile(Mtiling[i+k], Ktiling[j+l], &Aseed);
				        irregular_tiled_matrix_desc_set_data(&ddescA, ptr, i+k, j+l, Mtiling[i+k], Ktiling[j+l], 0, rank);
			        }


        for (i = ddescB.grid.rrank*ddescB.grid.strows; i < KT; i+=ddescB.grid.rows*ddescB.grid.strows)
	        for (k = 0; k < ddescB.grid.stcols; ++k)
		        for (j = ddescB.grid.crank*ddescB.grid.stcols; j < NT; j+=ddescB.grid.cols*ddescB.grid.stcols)
			        for (l = 0; l < ddescB.grid.stcols; ++l) {
				        ptr = init_tile(Ktiling[i+k], Ntiling[j+l], &Bseed);
				        irregular_tiled_matrix_desc_set_data(&ddescB, ptr, i+k, j+l, Ktiling[i+k], Ntiling[j+l], 0, rank);
			        }

        /* summa_zplrnt( dague, 0, (irregular_tiled_matrix_desc_t *)&ddescA, Aseed); */
        /* summa_zplrnt( dague, 0, (irregular_tiled_matrix_desc_t *)&ddescB, Bseed); */
        if(1 || loud > 2) printf("Done\n");

        /* Create DAGuE */
        SYNC_TIME_START();
        dague_handle_t* DAGUE_zsumma = summa_zsumma_New(tA, tB, alpha,
                                                        (irregular_tiled_matrix_desc_t*)&ddescA,
                                                        (irregular_tiled_matrix_desc_t*)&ddescB,
                                                        (irregular_tiled_matrix_desc_t*)&ddescC);

        dague_enqueue(dague, DAGUE_zsumma);
        if( loud > 2 ) SYNC_TIME_PRINT(rank, ("zsumma\tDAG created\n"));

        /* lets rock! */
        PASTE_CODE_PROGRESS_KERNEL(dague, zsumma);

        summa_zsumma_Destruct( DAGUE_zsumma );

        /* dague_data_free(ddescA.mat); */
        irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescA);
        /* dague_data_free(ddescB.mat); */
        irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescB);
    }
    else {
#if 0
	    int Am, An, Bm, Bn;
        PASTE_CODE_ALLOCATE_MATRIX(ddescC2, check,
            irregular_tiled_matrix_desc, (&ddescC2, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDC, N, 0, 0,
                                   M, N, SMB, SNB, P));

        dplasma_zplrnt( dague, 0, (irregular_tiled_matrix_desc_t *)&ddescC2, Cseed);

#if defined(PRECISION_z) || defined(PRECISION_c)
        for(tA=0; tA<3; tA++) {
            for(tB=0; tB<3; tB++) {
#else
        for(tA=0; tA<2; tA++) {
            for(tB=0; tB<2; tB++) {
#endif
                if ( trans[tA] == PlasmaNoTrans ) {
                    Am = M; An = K;
                } else {
                    Am = K; An = M;
                }
                if ( trans[tB] == PlasmaNoTrans ) {
                    Bm = K; Bn = N;
                } else {
                    Bm = N; Bn = K;
                }
                PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
                    irregular_tiled_matrix_desc, (&ddescA, tile_coll_ComplexDouble, tile_coll_Tile,
                                           nodes, rank, MB, NB, LDA, LDA, 0, 0,
                                           Am, An, SMB, SNB, P));
                PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
                    irregular_tiled_matrix_desc, (&ddescB, tile_coll_ComplexDouble, tile_coll_Tile,
                                           nodes, rank, MB, NB, LDB, LDB, 0, 0,
                                           Bm, Bn, SMB, SNB, P));

                dplasma_zplrnt( dague, 0, (irregular_tiled_matrix_desc_t *)&ddescA, Aseed);
                dplasma_zplrnt( dague, 0, (irregular_tiled_matrix_desc_t *)&ddescB, Bseed);

                if ( rank == 0 ) {
                    printf("***************************************************\n");
                    printf(" ----- TESTING ZGEMM (%s, %s) -------- \n",
                           transstr[tA], transstr[tB]);
                }

                /* matrix generation */
                if(loud) printf("Generate matrices ... ");
                dplasma_zlacpy( dague, PlasmaUpperLower,
                                (irregular_tiled_matrix_desc_t *)&ddescC2, (irregular_tiled_matrix_desc_t *)&ddescC );
                if(loud) printf("Done\n");

                /* Create SUMMA DAGuE */
                if(loud) printf("Compute ... ... ");
                summa_zsumma(dague, trans[tA], trans[tB],
                              (dague_complex64_t)alpha,
                              (irregular_tiled_matrix_desc_t *)&ddescA,
                              (irregular_tiled_matrix_desc_t *)&ddescB,
                              (irregular_tiled_matrix_desc_t *)&ddescC);
                if(loud) printf("Done\n");

                irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescA);
                irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescB);

                /* Check the solution */
                info_solution = check_solution( dague, (rank == 0) ? loud : 0,
                                                trans[tA], trans[tB],
                                                alpha, Am, An, Aseed,
                                                       Bm, Bn, Bseed,
                                                M,  N,
                                                &ddescC);
                if ( rank == 0 ) {
                    if (info_solution == 0) {
                        printf(" ---- TESTING ZSUMMA (%s, %s) ...... PASSED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    else {
                        printf(" ---- TESTING ZSUMMA (%s, %s) ... FAILED !\n",
                               transstr[tA], transstr[tB]);
                    }
                    printf("***************************************************\n");
                }
            }
        }
#if defined(_UNUSED_)
            }
        }
#endif
        irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescC2);
#endif
    }

    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescC);

    cleanup_dague(dague, iparam);

    return info_solution;
}

/**********************************
 * static functions
 **********************************/

/*------------------------------------------------------------------------
 *  Check the accuracy of the solution
 */
static int check_solution( dague_context_t *dague, int loud,
                           PLASMA_enum transA, PLASMA_enum transB,
                           dague_complex64_t alpha, int Am, int An, int Aseed,
                           int Bm, int Bn, int Bseed,
                           int M,  int N,
                           irregular_tiled_matrix_desc_t *ddescCfinal )
{
    int info_solution = 1;
    (void)dague; (void)loud; (void)transA; (void)transB;
    (void)alpha; (void)Am; (void)An; (void)Aseed; (void)Bm;
    (void)Bn; (void)Bseed; (void)M; (void)N; (void)ddescCfinal;

#if 0
    double Anorm, Bnorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm;
    double eps, result;
    int K  = ( transA == PlasmaNoTrans ) ? An : Am ;
    int MB = ddescCfinal->super.mb;
    int NB = ddescCfinal->super.nb;
    int LDA = Am;
    int LDB = Bm;
    int LDC = M;
    int rank  = ddescCfinal->super.myrank;

    eps = LAPACKE_dlamch_work('e');

    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1,
        irregular_tiled_matrix_desc, (&ddescA, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescB, 1,
        irregular_tiled_matrix_desc, (&ddescB, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDB, Bn, 0, 0,
                               Bm, Bn, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(ddescC, 1,
        irregular_tiled_matrix_desc, (&ddescC, matrix_ComplexDouble, matrix_Lapack,
                               1, rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1));

    dplasma_zplrnt( dague, 0, (irregular_tiled_matrix_desc_t *)&ddescA, Aseed );
    dplasma_zplrnt( dague, 0, (irregular_tiled_matrix_desc_t *)&ddescB, Bseed );
    dplasma_zplrnt( dague, 0, (irregular_tiled_matrix_desc_t *)&ddescC, Cseed );

    Anorm        = dplasma_zlange( dague, PlasmaInfNorm, (irregular_tiled_matrix_desc_t*)&ddescA );
    Bnorm        = dplasma_zlange( dague, PlasmaInfNorm, (irregular_tiled_matrix_desc_t*)&ddescB );
    Cinitnorm    = dplasma_zlange( dague, PlasmaInfNorm, (irregular_tiled_matrix_desc_t*)&ddescC );
    Cdplasmanorm = dplasma_zlange( dague, PlasmaInfNorm, (irregular_tiled_matrix_desc_t*)ddescCfinal );

    if ( rank == 0 ) {
        cblas_zgemm(CblasColMajor,
                    (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
                    M, N, K,
                    CBLAS_SADDR(alpha), ddescA.mat, LDA,
                                        ddescB.mat, LDB,
                    CBLAS_SADDR(beta),  ddescC.mat, LDC );
    }

    Clapacknorm = dplasma_zlange( dague, PlasmaInfNorm, (irregular_tiled_matrix_desc_t*)&ddescC );

    dplasma_zgeadd( dague, PlasmaNoTrans, -1.0, (irregular_tiled_matrix_desc_t*)ddescCfinal,
                                           1.0, (irregular_tiled_matrix_desc_t*)&ddescC );

    Rnorm = dplasma_zlange( dague, PlasmaMaxNorm, (irregular_tiled_matrix_desc_t*)&ddescC);

    if ( rank == 0 ) {
        if ( loud > 2 ) {
            printf("  ||A||_inf = %e, ||B||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*B+b*C)||_inf = %e, ||dplasma(a*A*B+b*C)||_inf = %e, ||R||_m = %e\n",
                   Anorm, Bnorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm);
        }

        result = Rnorm / ((Anorm + Bnorm + Cinitnorm) * max(M,N) * eps);
        if (  isinf(Clapacknorm) || isinf(Cdplasmanorm) ||
              isnan(result) || isinf(result) || (result > 10.0) ) {
            info_solution = 1;
        }
        else {
            info_solution = 0;
        }
    }

#if defined(DAGUE_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescA);
    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescB);
    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescC);
#endif
    return info_solution;
}
