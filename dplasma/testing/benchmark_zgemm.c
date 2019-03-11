/*
 * Copyright (c) 2019      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec.h"
#include "parsec/execution_stream.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/profiling.h"
#include "parsec/utils/mca_param.h"

#include "common.h"
#include "parsec/data_dist/matrix/irregular_tiled_matrix.h"
#include "dplasma_z.h"
#include "flops.h"
#include "dplasma/lib/irregular_tiled_matrix_init.h"
#include "dplasma/lib/zgemm_benchmark.h"
#include "dplasma/include/dplasmatypes.h"

//static unsigned long long int Rnd64seed = 100;
#define Rnd64_A  6364136223846793005ULL
#define Rnd64_C  1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20
#define EPSILON  0.000001L

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    unsigned long long int Aseed = 3872;
    unsigned long long int Bseed = 4674;
    unsigned long long int Cseed = 4242;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
    iparam[IPARAM_NB] = 100;
    iparam[IPARAM_LDB] = 7;
#if defined(PARSEC_HAVE_CUDA) && 1
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);

    if(0) {
      volatile int loop = 1;
      char hostname[64];
      gethostname(hostname, 64);
      fprintf(stderr, "on %s gdb -p %p\n", hostname, getpid());
      while(loop) {
        sleep(1);
      }
    }

    int rank  = iparam[IPARAM_RANK];
    int nodes = iparam[IPARAM_NNODES];
    int M     = iparam[IPARAM_MB];
    int N     = iparam[IPARAM_NB];
    int CN;
    int MB    = iparam[IPARAM_M];
    int NB    = iparam[IPARAM_N];
    int KB    = iparam[IPARAM_K];
    int loud  = iparam[IPARAM_VERBOSE];
    int benchmode = iparam[IPARAM_LDB];

    printf("benchmode = %x\n", benchmode);
    
    if( benchmode > 7 || benchmode <= 0 ) {
        fprintf(stderr, "Incorrect benchmark mode ('-B %d'): Possible values are a bitwise or between 1 (do GPU computations), 2 (do CPU to GPU transfers) and 4 (do GPU to CPU transfers)\n", benchmode);
        exit(2);
    }

    if( nodes > 1 ) {
        fprintf(stderr, "Incorrect run for the benchmark: this benchmark only runs single node, to test the performance of a GPU GEMM\n");
        exit(2);
    }
    
    if( (benchmode & 0x2) || (benchmode & 0x4) ) {
        CN = N * NB;
    } else {
        CN = NB;
    }
    
    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, M * MB, CN, 0, 0,
                               M * MB, CN, 1, 1, 1));

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, KB, M * MB, KB, 0, 0,
                               M * MB, KB, 1, 1, 1));
    PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
        two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, KB, NB, M * KB, NB, 0, 0,
                               M * KB, NB, 1, 1, 1));

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, Bseed);
    dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC, Cseed);
    if(loud > 2) printf("Done\n");

    double gflops = -1.0, flops = FLOPS_ZGEMM((DagDouble_t)MB,(DagDouble_t)NB,(DagDouble_t)KB) * N * M;

    PASTE_MKL_WARMUP();

    /* Create Parsec taskpool */
    for(int run = 0; run < 20; run++) {
        parsec_devices_release_memory();
        
        parsec_taskpool_t* PARSEC_zgemm_bench = parsec_zgemm_benchmark_new((const parsec_tiled_matrix_dc_t *)&dcA,
                                                                           (const parsec_tiled_matrix_dc_t *)&dcB,
                                                                           (parsec_tiled_matrix_dc_t *)&dcC,
                                                                           M, N);
        parsec_dgemm_benchmark_taskpool_t* zgemm_bench = (parsec_dgemm_benchmark_taskpool_t*)PARSEC_zgemm_bench;
        
        dplasma_add2arena_tile(zgemm_bench->arenas[PARSEC_zgemm_benchmark_DEFAULT_ARENA],
                               MB*NB*sizeof(parsec_complex64_t),
                               PARSEC_ARENA_ALIGNMENT_SSE,
                               parsec_datatype_double_complex_t, MB);
        zgemm_bench->_g_mode = benchmode;
        
        parsec_enqueue(parsec, PARSEC_zgemm_bench);

        if( loud > 2 ) SYNC_TIME_PRINT(rank, ("zbench\tDAG created\n"));
        
        /* lets rock! */
        SYNC_TIME_START();
        parsec_context_start(parsec);
        parsec_context_wait(parsec);
        SYNC_TIME_PRINT(rank, ("ZGEMM_BENCHMARK\tDO_COMPUTE=%d DO_CPU_TO_GPU=%d DO_GPU_TO_CPU=%d M=%d N=%d K=%d nb_gemm_in_sequence=%d nb_parallel_sequences=%d: %14f gflops\n",
                               (benchmode & 0x1) ? 1 : 0, (benchmode & 0x2) ? 1 : 0, (benchmode & 0x4) ? 1 : 0, MB, NB, KB, N, M,
                               gflops=(flops/1e9)/sync_time_elapsed));
        
        if (zgemm_bench->arenas[PARSEC_zgemm_benchmark_DEFAULT_ARENA])
            parsec_matrix_del2arena( zgemm_bench->arenas[PARSEC_zgemm_benchmark_DEFAULT_ARENA] );
        parsec_taskpool_free(PARSEC_zgemm_bench);

        //parsec_devices_dump_and_reset_statistics(parsec);
        parsec_devices_reset_load(parsec);
    }

    if(iparam[IPARAM_HNB] != iparam[IPARAM_NB])
        parsec_taskpool_sync_ids(); /* recursive DAGs are not synchronous on ids */

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
    parsec_data_free(dcB.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);
        
    cleanup_parsec(parsec, iparam);

    return info_solution;
}
