/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    int Aseed = 3872;
    int Bseed = 4674;
    int Cseed = 2873;
    int tA = PlasmaNoTrans;
    int tB = PlasmaNoTrans;
    parsec_complex64_t alpha =  0.51;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
    beta  += I * 0.21;
#endif

    /* Set defaults for non argv iparams */
    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);
#if defined(PARSEC_HAVE_CUDA) && 1
    iparam[IPARAM_NGPUS] = 0;
#endif
    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);

    PASTE_CODE_FLOPS(FLOPS_ZGEMM, ((DagDouble_t)M,(DagDouble_t)N,(DagDouble_t)K));

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        two_dim_block_cyclic, (&dcC, matrix_ComplexDouble, matrix_Tile,
                               nodes, rank, MB, NB, LDC, N, 0, 0,
                               M, N, SMB, SNB, P));

    /* initializing matrix structure */
    if(!check)
    {
        PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDA, K, 0, 0,
                                   M, K, SMB, SNB, P));
        PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
            two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                   nodes, rank, MB, NB, LDB, N, 0, 0,
                                   K, N, SMB, SNB, P));

        /* matrix generation */
        if(loud > 2) printf("+++ Generate matrices ... ");
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcA, Aseed);
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, Bseed);
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcC, Cseed);
        if(loud > 2) printf("Done\n");

        int t;
        for ( t = 0; t < 1; ++t) {
            /* Create PaRSEC */
            PASTE_CODE_ENQUEUE_KERNEL(parsec, zgemm_summit,
                                      (tA, tB, alpha,
                                       (parsec_tiled_matrix_dc_t *)&dcA,
                                       (parsec_tiled_matrix_dc_t *)&dcB,
                                       (parsec_tiled_matrix_dc_t *)&dcC,
                                       sB, sC, sD, P, Q));

            /* lets rock! */
            PASTE_CODE_PROGRESS_KERNEL(parsec, zgemm);

            dplasma_zgemm_summit_Destruct( PARSEC_zgemm_summit );
        }

        parsec_data_free(dcA.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcA);
        parsec_data_free(dcB.mat);
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcB);
    }
    else {
        printf("testing not coded yet\n");
    }

    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)&dcC);

    cleanup_parsec(parsec, iparam);

    return info_solution;
}
