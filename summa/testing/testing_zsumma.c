/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "irregular_tiled_matrix.h"
#include "summa_z.h"
#include "flops.h"

//static unsigned long long int Rnd64seed = 100;
#define Rnd64_A  6364136223846793005ULL
#define Rnd64_C  1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20
#define EPSILON  0.000001L

static void init_tiling(unsigned int *T, unsigned long long int *seed, int MT, int MB, int M)
{
	int t;
	(void)seed;

	for (t = 0; t < MT; ++t) T[t] = MB;
	if (M%MB != 0) T[MT-1] = M%MB;
	/* good old regular tiling with smaller last tile */

	if (MT > 1) {
		/* unsigned int share = (MB/10 > 0) ? MB/10 : 1; */
		/* T[0] += share*(MT-1); */
		/* for (t = 1; t < MT; ++t) T[t] -= share; */
	}
#if defined(SUMMA_WITH_RANDOM_TILING)
	int p;
	unsigned int lower_bound = (MB/2 == 0)? 1: MB/2;
	unsigned int upper_bound = MB*2;
	unsigned long long int ran = *seed;
	unsigned int share = (MB/10 > 0) ? MB/10 : 1;

	for (p = 0; p < MT*MT/2; ++p) {
		int t1 = ran%MT;
		ran = Rnd64_A * ran + Rnd64_C;
		int t2 = t1;
		while (t2 == t1) {
			t2 = ran%MT;
			ran = Rnd64_A * ran + Rnd64_C -1;
		}

		/* steal 1 from t1, give it to t2 if the boundaries are respected */
		if (T[t1] > lower_bound && T[t2] < upper_bound) {
			T[t1] -= share;
			T[t2] += share;
		}
	}
    *seed = ran;
#endif
}

static void* init_tile(int mb, int nb, unsigned long long int *seed)
{
    unsigned long long int ran = *seed;
    int i, j;

	/* fprintf(stdout, "Allocating %dx%d*sizeof(%d)\n", mb, nb, sizeof(parsec_complex64_t)); */
	parsec_complex64_t *array = (parsec_complex64_t*)malloc(sizeof(parsec_complex64_t)*mb*nb);

    for (j = 0; j < nb; ++j)
	    for (i = 0; i < mb; ++i) {
		    array[i+j*mb] = ran%10;
            ran = Rnd64_A * ran + Rnd64_C;
#if defined(PRECISION_z) || defined(PRECISION_c)
            array[i+j*mb] += I*(ran%10);
            ran = Rnd64_A * ran + Rnd64_C;
#endif
        }
    *seed = ran;
    return array;
}

static void init_random_matrix(irregular_tiled_matrix_desc_t *M, unsigned long long int seed, parsec_complex64_t **storage_map)
{
	void *ptr;
	int i, j, k, l, u = 0;
	for (i = 0; i < M->mt; i+=M->grid.strows)
		for (k = 0; k < M->grid.stcols && (i+k) < M->mt; ++k)
			for (j = 0; j < M->nt; j+=M->grid.stcols)
				for (l = 0; l < M->grid.stcols && (j+l)<M->nt; ++l) {
					unsigned int rank = tile_owner(i+k,j+l,&M->grid);
					ptr = (rank == ((parsec_ddesc_t*)M)->myrank) ? init_tile(M->max_mb, M->max_tile/M->max_mb, &seed) : NULL;

					if (ptr) u++;
					uint32_t idx = ((parsec_ddesc_t*)M)->data_key((parsec_ddesc_t*)M, i+k, j+l);
					irregular_tiled_matrix_desc_set_data(M, ptr, idx, M->Mtiling[i+k], M->Ntiling[j+l], 0, rank);
					storage_map[idx] = ptr;
				}


	fprintf(stdout, "Allocated %d blocks of size %dx%d\n", u, M->max_mb, M->max_tile/M->max_mb);
}

static void init_empty_matrix(irregular_tiled_matrix_desc_t *M, parsec_complex64_t **storage_map)
{
	int i, j, k, l, u = 0;
	void *ptr;
	/* i progresses by strows steps, while k progresses one by one from 0 to strows-1 */
	for (i = 0; i < M->mt; i+=M->grid.strows)
		for (k = 0; k < M->grid.stcols && (i+k) < M->mt; ++k)
			/* j progresses by stcols steps, while l progresses one by one from 0 to stcols-1 */
			for (j = 0; j < M->nt; j+=M->grid.stcols)
				for (l = 0; l < M->grid.stcols && (j+l) < M->nt; ++l) {
					unsigned int rank = tile_owner(i+k,j+l,&M->grid);
					/* Workaround void *ptr = calloc(M->Mtiling[i+k]*M->Ntiling[j+l], sizeof(parsec_complex64_t)); */
					ptr = (rank == ((parsec_ddesc_t*)M)->myrank) ? calloc(M->max_tile, sizeof(parsec_complex64_t)) : NULL;
					if (ptr) u++;
					uint32_t idx = ((parsec_ddesc_t*)M)->data_key((parsec_ddesc_t*)M, i+k, j+l);
					irregular_tiled_matrix_desc_set_data(M, ptr, idx, M->Mtiling[i+k], M->Ntiling[j+l], 0, rank);
					storage_map[idx] = ptr;
				}


	fprintf(stdout, "Allocated %d blocks of size %d\n", u, M->max_tile);
}

static void fini_matrix(parsec_complex64_t **Mstorage, int nb)
{
	int i;
	for (i = 0; i < nb; ++i)
		free(Mstorage[i]);
}

static void copy_tile_in_matrix(parsec_ddesc_t* M, parsec_complex64_t *check)
{
	irregular_tiled_matrix_desc_t *descM = (irregular_tiled_matrix_desc_t*)M;
	int i, j, k, ipos, jpos;
	ipos = 0;
	for (i = 0; i < descM->mt; ++i) {
		jpos = 0;
		for (j = 0; j < descM->nt; ++j) {
			parsec_data_t *t_ij = M->data_of(M, i, j);
			irregular_tile_data_copy_t *ct_ij = (irregular_tile_data_copy_t*)t_ij->device_copies[0];
			parsec_complex64_t *ptr = ((parsec_data_copy_t*)ct_ij)->device_private;
			int ct_ij_nb = descM->Ntiling[j];
			int ct_ij_mb = descM->Mtiling[i];
			for (k = 0; k < ct_ij_nb; ++k) {
				/* copy each column of tile ij at the right position in M */
				memcpy(check+(ipos+(jpos+k)*descM->lm),
				       ptr+k*ct_ij_mb,
				       ct_ij_mb*sizeof(parsec_complex64_t));
			}
			/* move the column cursor to the next tile */
			jpos += descM->Ntiling[j];
		}
		/* move the row cursor to the next tile */
		ipos += descM->Mtiling[i];
	}
}

#if defined(PARSEC_DEBUG_PARANOID)
static void print_matrix_data(irregular_tiled_matrix_desc_t* A, const char *Aid, parsec_complex64_t* checkA)
{
#if defined(PRECISION_z)
#define FORMAT " %f+i%f%s"
#elif defined(PRECISION_c)
#define FORMAT " %lf+i%lf%s"
#elif defined(PRECISION_d)
#define FORMAT " %lf%s"
#else
#define FORMAT " %f%s"
#endif

#if defined(PRECISION_z) || defined(PRECISION_c)
#define cmplx_print(z) creal(z), cimag(z)
#else
#define cmplx_print(z) (z)
#endif

	/* print the matrix in scilab-friendly-ready-to-c/c format */
	int i, j;
	fprintf(stdout, "Matrix_%s = [\n", Aid);
	for (i = 0; i < A->m; i++)
		for (j = 0; j < A->n; ++j)
			fprintf(stdout, FORMAT, cmplx_print(checkA[i+A->m*j]),
			        (j!=A->n-1)?",":(i!=A->m-1)?";\n":"];\n");
}
#endif

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

static void check_solution(irregular_tiled_matrix_desc_t *ddescA, int tA, parsec_complex64_t alpha,
                           irregular_tiled_matrix_desc_t *ddescB, int tB,
                           irregular_tiled_matrix_desc_t *ddescC,
                           int M, int N, int K)
{
	int tempmm = M, tempnn = N, tempkk = K;
	int lda = M, ldb = K, ldc = M;
	int i, b = 1;
	parsec_complex64_t *checkA, *checkB, *checkC;
	parsec_complex64_t beta = (parsec_complex64_t)-1.0;

	checkA = (parsec_complex64_t*)calloc(M*K,sizeof(parsec_complex64_t));
	checkB = (parsec_complex64_t*)calloc(K*N,sizeof(parsec_complex64_t));
	checkC = (parsec_complex64_t*)calloc(M*N,sizeof(parsec_complex64_t));

	fprintf(stdout, "+++ Checking solution .");
	copy_tile_in_matrix((parsec_ddesc_t*)ddescA, checkA);
	copy_tile_in_matrix((parsec_ddesc_t*)ddescB, checkB);
	copy_tile_in_matrix((parsec_ddesc_t*)ddescC, checkC);
	fprintf(stdout, ".");
#if defined(PARSEC_DEBUG_PARANOID)
	print_matrix_data(ddescA, "A", checkA);
	print_matrix_data(ddescB, "B", checkB);
	print_matrix_data(ddescC, "C", checkC);
#endif

	CORE_zgemm(tA, tB, tempmm, tempnn, tempkk, alpha, checkA, lda, checkB, ldb, beta, checkC, ldc);
	fprintf(stdout, ".");

#if defined(PARSEC_DEBUG_PARANOID)
	fprintf(stdout, "D = A * B - C (D should be null)\n");
	print_matrix_data(ddescC, "D", checkC);
#endif

	for (i = 0; i < M*N; ++i)
		if (cabs(checkC[i]) > EPSILON) {
			b = 0;
			break;
		}

	if (!b) fprintf(stdout, " test FAILED: C is not null enough!\n");
	else    fprintf(stdout, " test SUCCEED: C is null!\n");

	free(checkA);
	free(checkB);
	free(checkC);
}



int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    unsigned long long int Aseed = 3872;
    unsigned long long int Bseed = 4674;
    unsigned long long int Tseed = 4242;
    int tA = PlasmaNoTrans;
    int tB = PlasmaNoTrans;
    parsec_complex64_t alpha = 1.;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(PARSEC_HAVE_CUDA) && 1
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize Parsec */
    parsec = setup_parsec(argc, argv, iparam);

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

    double gflops = -1.0, flops = FLOPS_ZSUMMA((DagDouble_t)M,(DagDouble_t)N,(DagDouble_t)K);

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

	if (rank == 0) {
		int i;
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
	}

    /* initializing matrix structure */
    irregular_tiled_matrix_desc_t ddescA;
    irregular_tiled_matrix_desc_init(&ddescA, tile_coll_ComplexDouble,
                                     nodes, rank, M, K, MT, KT,
                                     Mtiling, Ktiling,
                                     0, 0, MT, KT, P, NULL);
    irregular_tiled_matrix_desc_t ddescB;
    irregular_tiled_matrix_desc_init(&ddescB, tile_coll_ComplexDouble,
                                     nodes, rank, K, N, KT, NT,
                                     Ktiling, Ntiling,
                                     0, 0, KT, NT, P, NULL);
    irregular_tiled_matrix_desc_t ddescC;
    irregular_tiled_matrix_desc_init(&ddescC, tile_coll_ComplexDouble,
                                     nodes, rank, M, N, MT, NT,
                                     Mtiling, Ntiling,
                                     0, 0, MT, NT, P, NULL);

    unsigned int max_tile = summa_imax(ddescA.max_tile, summa_imax(ddescB.max_tile, ddescC.max_tile));
    unsigned int max_mb = summa_imax(ddescA.max_mb, summa_imax(ddescB.max_mb, ddescC.max_mb));
    ddescA.max_tile = ddescB.max_tile = ddescC.max_tile = max_tile;
    ddescA.max_mb = ddescB.max_mb = ddescC.max_mb = max_mb;

	fprintf(stdout, "max_tile=%d, max_mb=%d\n", max_tile, max_mb);
	parsec_complex64_t **Astorage = (parsec_complex64_t**)calloc(MT*KT, sizeof(parsec_complex64_t*));
	parsec_complex64_t **Bstorage = (parsec_complex64_t**)calloc(KT*NT, sizeof(parsec_complex64_t*));
	parsec_complex64_t **Cstorage = (parsec_complex64_t**)calloc(MT*NT, sizeof(parsec_complex64_t*));

	/* matrix generation */
	if (loud > 2) printf("+++ Generate matrices ... ");
	init_random_matrix(&ddescA, Aseed, Astorage);
	init_random_matrix(&ddescB, Bseed, Bstorage);
	init_empty_matrix(&ddescC, Cstorage);
	if(loud > 2) printf("Done\n");

#if 0
	if (rank == 0)
		export_pythons(&ddescA, &ddescB, &ddescC, P, Q, N, NB, MB, nodes);
#endif

	free(Mtiling);
    free(Ntiling);
    free(Ktiling);

#if defined(PARSEC_DEBUG_NOISIER)
    fprintf(stdout, "Matrix A:\n");
    print_matrix_meta(&ddescA);
    fprintf(stdout, "Matrix B:\n");
    print_matrix_meta(&ddescB);
    fprintf(stdout, "Matrix C:\n");
    print_matrix_meta(&ddescC);
#endif

	double A = 1, B = 2, C = 0;
	CORE_zgemm(PlasmaNoTrans, PlasmaNoTrans,
			   1, 1, 1, 3., &A, 1, &B, 1, 1., &C, 1);

    /* Create Parsec handle */
    SYNC_TIME_START();
    parsec_handle_t* PARSEC_zsumma = summa_zsumma_New(tA, tB, alpha,
                                                    (irregular_tiled_matrix_desc_t*)&ddescA,
                                                    (irregular_tiled_matrix_desc_t*)&ddescB,
                                                    (irregular_tiled_matrix_desc_t*)&ddescC);

#if defined(PARSEC_HAVE_RECURSIVE)
    if(iparam[IPARAM_HNB] != iparam[IPARAM_NB])
		summa_zsumma_setrecursive(PARSEC_zsumma, iparam[IPARAM_HNB], iparam[IPARAM_HNB]);
#endif
	
    parsec_enqueue(parsec, PARSEC_zsumma);
    if( loud > 2 ) SYNC_TIME_PRINT(rank, ("zsumma\tDAG created\n"));

    /* lets rock! */
    SYNC_TIME_START();
    TIME_START();
    parsec_context_wait(parsec);
    SYNC_TIME_PRINT(rank, ("ZSUMMA\tPxQ= %3d %-3d NB= %4d N= %7d : %14f gflops\n",
                           P, Q, NB, N,
                           gflops=(flops/1e9)/sync_time_elapsed));

	summa_zsumma_Destruct( PARSEC_zsumma );

	if(iparam[IPARAM_HNB] != iparam[IPARAM_NB])
		parsec_handle_sync_ids(); /* recursive DAGs are not synchronous on ids */

	if (check)
	    check_solution(&ddescA, tA, alpha, &ddescB, tB, &ddescC, M, N, K);

	fini_matrix(Astorage, MT*KT);
	fini_matrix(Bstorage, KT*NT);
	fini_matrix(Cstorage, MT*NT);

	free(Astorage);
	free(Bstorage);
	free(Cstorage);

    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescA);
    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescB);
    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)&ddescC);

    cleanup_parsec(parsec, iparam);

    return info_solution;
}
