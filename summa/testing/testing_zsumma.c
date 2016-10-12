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

static void init_tiling(unsigned int *T, unsigned long long int *seed, int MT, int MB, int M)
{
	int t, p;
	unsigned long long int ran = *seed;

	for (t = 0; t < MT; ++t) T[t] = MB;
	if (M%MB != 0) T[MT-1] = M%MB;
	/* good old regular tiling with smaller last tile */

	unsigned int lower_bound = (MB*3)/5;
	unsigned int upper_bound = (MB*8)/5;
	for (p = 0; p < 5*MT; ++p) {
		int t1 = ran%MT;
		ran = Rnd64_A * ran +Rnd64_C;
		int t2 = t1;
		while (t2 ==t1) {
			t2 = ran%MT;
			ran = Rnd64_A * ran + Rnd64_C;
		}

		/* steal 1 from t1, give it to t2 if the boundaries are respected */
		if (T[t1] > lower_bound && T[t2] < upper_bound) {
			T[t1]--;
			T[t2]++;
		}
	}
    *seed = ran;
}

static void* init_tile(int mb, int nb, unsigned long long int *seed)
{
    unsigned long long int ran = *seed;
    int i, j;
    dague_complex64_t *array = (dague_complex64_t*)malloc(sizeof(dague_complex64_t)*mb*nb);

    for (j = 0; j < nb; ++j)
	    for (i = 0; i < mb; ++i) {
		    array[i+j*mb] = ran%10;
            ran = Rnd64_A * ran + Rnd64_C;
        }
    *seed = ran;
    return array;
}

static void init_random_matrix(irregular_tiled_matrix_desc_t* M, unsigned long long int seed)
{
	int i, j, k, l;
	int rank = M->grid.rank;
	for (i = M->grid.rrank*M->grid.strows; i < M->mt; i+=M->grid.rows*M->grid.strows)
		for (k = 0; k < M->grid.stcols; ++k)
			for (j = M->grid.crank*M->grid.stcols; j < M->nt; j+=M->grid.cols*M->grid.stcols)
				for (l = 0; l < M->grid.stcols; ++l) {
					void *ptr = init_tile(M->Mtiling[i+k], M->Ntiling[j+l], &seed);
					irregular_tiled_matrix_desc_set_data(M, ptr, i+k, j+l, M->Mtiling[i+k], M->Ntiling[j+l], 0, rank);
				}
}

static void init_empty_matrix(irregular_tiled_matrix_desc_t* M)
{
	int i, j, k, l;
	int rank = M->grid.rank;
	for (i = M->grid.rrank*M->grid.strows; i < M->mt; i+=M->grid.rows*M->grid.strows)
        for (k = 0; k < M->grid.stcols; ++k)
            for (j = M->grid.crank*M->grid.stcols; j < M->nt; j+=M->grid.cols*M->grid.stcols)
                for (l = 0; l < M->grid.stcols; ++l) {
                    void *ptr = calloc(M->Mtiling[i+k]*M->Ntiling[j+l], sizeof(dague_complex64_t));
                    irregular_tiled_matrix_desc_set_data(M, ptr, i+k, j+l, M->Mtiling[i+k], M->Ntiling[j+l], 0, rank);
                }
}

static void copy_tile_in_matrix(dague_ddesc_t* M, dague_complex64_t *check)
{
	irregular_tiled_matrix_desc_t *descM = (irregular_tiled_matrix_desc_t*)M;
	int i, j, k, ipos, jpos;
	ipos = 0;
	for (i = 0; i < descM->mt; ++i) {
		jpos = 0;
		for (j = 0; j < descM->nt; ++j) {
			dague_data_t *t_ij = M->data_of(M, i, j);
			irregular_tile_data_copy_t *ct_ij = (irregular_tile_data_copy_t*)t_ij->device_copies[0];
			dague_complex64_t *ptr = ((dague_data_copy_t*)ct_ij)->device_private;
			for (k = 0; k < ct_ij->nb; ++k) {
				/* copy each column of tile ij at the right position in M */
				memcpy(check+(ipos+(jpos+k)*descM->lm),
				       ptr+k*ct_ij->mb,
				       ct_ij->mb*sizeof(dague_complex64_t));
			}
			/* move the column cursor to the next tile */
			jpos += descM->Ntiling[j];
		}
		/* move the row cursor to the next tile */
		ipos += descM->Mtiling[i];
	}
}

static void print_matrix_data(irregular_tiled_matrix_desc_t* A, const char *Aid, dague_complex64_t* checkA)
{
	/* print the matrix in scilab-friendly-ready-to-c/c format */
	int i, j;
	fprintf(stdout, "Matrix_%s = [\n", Aid);
	for (i = 0; i < A->m; i++)
		for (j = 0; j < A->n; ++j)
			fprintf(stdout, " %f%s", checkA[i+A->m*j],
			        (j!=A->n-1)?",":(i!=A->m-1)?";\n":"];\n");
}

/* prints meta deta of the matrix */
static void print_matrix_meta(irregular_tiled_matrix_desc_t* A)
{
    fprintf(stdout, "  Grid: %dx%d\n",A->grid.rows, A->grid.cols);
    fprintf(stdout, "  M=%d, N=%d, MT=%d, NT=%d\n", A->m, A->n, A->mt, A->nt);

    int i;
    fprintf(stdout, "  M tiling:");
    for (i = 0; i < A->mt; ++i) fprintf(stdout, " %d", A->Mtiling[i]);
    fprintf(stdout, "\n");
    fprintf(stdout, "  N tiling:");
    for (i = 0; i < A->nt; ++i) fprintf(stdout, " %d", A->Ntiling[i]);
    fprintf(stdout, "\n");

    fprintf(stdout, "  i=%d, j=%d, nb_local_tiles=%d\n", A->i, A->j, A->nb_local_tiles);
    fprintf(stdout, "  lm=%d, ln=%d, lmt=%d, lnt=%d\n", A->lm, A->ln, A->lmt, A->lnt);
}







int main(int argc, char ** argv)
{
	int i;
	dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    unsigned long long int Aseed = 3872;
    unsigned long long int Bseed = 4674;
    unsigned long long int Tseed = 4242;
    int tA = PlasmaNoTrans;
    int tB = PlasmaNoTrans;
    dague_complex64_t alpha = 1.;

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
    (void)cores;(void)gpus;(void)P;(void)Q;(void)M;(void)N;(void)K;(void)NRHS;
    (void)LDA;(void)LDB;(void)LDC;(void)IB;(void)MB;(void)NB;(void)MT;(void)NT;(void)KT;
    (void)SMB;(void)SNB;(void)HMB;(void)HNB;(void)check;(void)loud;(void)async;
    (void)scheduler;(void)butterfly_level;(void)check_inv;(void)random_seed;(void)matrix_init;

    PASTE_CODE_FLOPS(FLOPS_ZSUMMA, ((DagDouble_t)M,(DagDouble_t)N,(DagDouble_t)K));

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    unsigned int *Mtiling = (unsigned int*)malloc(MT*sizeof(unsigned int));
    unsigned int *Ktiling = (unsigned int*)malloc(KT*sizeof(unsigned int));
    unsigned int *Ntiling = (unsigned int*)malloc(NT*sizeof(unsigned int));

    int KB = 1+(K-1)/KT;

    init_tiling(Mtiling, &Tseed, MT, MB, M);
    init_tiling(Ntiling, &Tseed, NT, NB, N);
    init_tiling(Ktiling, &Tseed, KT, KB, K);

#if defined(DAGUE_DEBUG_NOISIER)
    fprintf(stdout, "(MT = %d, mean(MB) = %d) x (KT = %d, mean(KB) = %d) x (NT = %d, mean(NB) = %d)\n",
            MT, MB, KT, KB, NT, NB);
    fprintf(stdout, "M tiling:");
    for (i = 0; i < MT; ++i) fprintf(stdout, " %d", Mtiling[i]);
    fprintf(stdout, "\n");
    fprintf(stdout, "K tiling:");
    for (i = 0; i < KT; ++i) fprintf(stdout, " %d", Ktiling[i]);
    fprintf(stdout, "\n");
    fprintf(stdout, "N tiling:");
    for (i = 0; i < NT; ++i) fprintf(stdout, " %d", Ntiling[i]);
    fprintf(stdout, "\n");
#endif

    dague_complex64_t *checkA, *checkB, *checkC;
    if (check) {
	    checkA      = (dague_complex64_t*)calloc(M*K,sizeof(dague_complex64_t));
	    checkB      = (dague_complex64_t*)calloc(K*N,sizeof(dague_complex64_t));
	    checkC      = (dague_complex64_t*)calloc(M*N,sizeof(dague_complex64_t));
    }


    /* initializing matrix structure */
    irregular_tiled_matrix_desc_t ddescA;
    irregular_tiled_matrix_desc_init(&ddescA, tile_coll_ComplexDouble,
                                     nodes, rank, M, K, MT, KT,
                                     Mtiling, Ktiling,
                                     0, 0, MT, KT, P);
    irregular_tiled_matrix_desc_t ddescB;
    irregular_tiled_matrix_desc_init(&ddescB, tile_coll_ComplexDouble,
                                     nodes, rank, K, N, KT, NT,
                                     Ktiling, Ntiling,
                                     0, 0, KT, NT, P);
    irregular_tiled_matrix_desc_t ddescC;
    irregular_tiled_matrix_desc_init(&ddescC, tile_coll_ComplexDouble,
                                     nodes, rank, M, N, MT, NT,
                                     Mtiling, Ntiling,
                                     0, 0, MT, NT, P);

    free(Mtiling);
    free(Ntiling);
    free(Ktiling);

    /* matrix generation */
    if(1 || loud > 2) printf("+++ Generate matrices ... ");
    init_random_matrix(&ddescA, Aseed);
    init_random_matrix(&ddescB, Bseed);
    init_empty_matrix(&ddescC);
    if(1 || loud > 2) printf("Done\n");

#if defined(DAGUE_DEBUG_NOISIER)
    fprintf(stdout, "Matrix A:\n");
    print_matrix_meta(&ddescA);
    fprintf(stdout, "Matrix B:\n");
    print_matrix_meta(&ddescB);
    fprintf(stdout, "Matrix C:\n");
    print_matrix_meta(&ddescC);
#endif

    /* Create DAGuE */
    dague_handle_t* DAGUE_zsumma = summa_zsumma_New(tA, tB, alpha,
                                                    (irregular_tiled_matrix_desc_t*)&ddescA,
                                                    (irregular_tiled_matrix_desc_t*)&ddescB,
                                                    (irregular_tiled_matrix_desc_t*)&ddescC);

    dague_enqueue(dague, DAGUE_zsumma);
    if( loud > 2 ) SYNC_TIME_PRINT(rank, ("zsumma\tDAG created\n"));

    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    dague_context_wait(dague);
    SYNC_TIME_PRINT(rank, ("ZSUMMA\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                           P, Q, NB, N,
                           gflops=(flops/1e9)/sync_time_elapsed));

    summa_zsumma_Destruct( DAGUE_zsumma );

    if (check) {
	    copy_tile_in_matrix((dague_ddesc_t*)&ddescA, checkA);
	    copy_tile_in_matrix((dague_ddesc_t*)&ddescB, checkB);
	    copy_tile_in_matrix((dague_ddesc_t*)&ddescC, checkC);
#if defined(DAGUE_DEBUG_PARANOID)
	    print_matrix_data(&ddescA, "A", checkA);
	    print_matrix_data(&ddescB, "B", checkB);
	    print_matrix_data(&ddescC, "C", checkC);
#endif
	    /* big dgemm time */
	    /* propagate transA, transB, column major */
	    int tempmm = M;
	    int tempnn = N;
	    int tempkk = K;
	    int lda = M;
	    int ldb = K;
	    int ldc = M;
	    dague_complex64_t beta = (dague_complex64_t)-1.0;
	    CORE_zgemm(tA, tB, tempmm, tempnn, tempkk, alpha, checkA, lda, checkB, ldb, beta, checkC, ldc);

#if defined(DAGUE_DEBUG_PARANOID)
	    fprintf(stdout, "D = A * B - C (D should be null)\n");
	    print_matrix_data(&ddescC, "D", checkC);
#endif

	    free(checkA);
	    free(checkB);
	    free(checkC);
    }

    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescA);
    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescB);
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
