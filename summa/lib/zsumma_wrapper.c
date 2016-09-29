/*
 * Copyright (c) 2016      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
/* #include "data_dist/matrix/two_dim_rectangle_cyclic.h" */

#include "irregular_tiled_matrix.h"

#include "zgemm_NN.h"
#include "zgemm_NT.h"
#include "zgemm_TN.h"
#include "zgemm_TT.h"

/**
 *******************************************************************************
 *
 * @ingroup summa_dgemm
 *
 *  dplasma_zgemm_New - Generates the handle that performs one of the following
 *  matrix-matrix operations. WARNING: The computations are not done by this call.
 *
 *    \f[ C = \alpha [op( A )\times op( B )],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha is scalar, and A, B and C are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   A is not transposed;
 *          = PlasmaTrans:     A is transposed;
 *          = PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   B is not transposed;
 *          = PlasmaTrans:     B is transposed;
 *          = PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta and is always considered null
 *
 * @param[out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C contain the matrix (
 *          alpha*op( A )*op( B ) )
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgemm_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm
 * @sa dplasma_zgemm_Destruct
 * @sa dplasma_cgemm_New
 * @sa dplasma_dgemm_New
 * @sa dplasma_sgemm_New
 *
 ******************************************************************************/
dague_handle_t*
summa_zgemm_New( PLASMA_enum transA, PLASMA_enum transB,
                 dague_complex64_t alpha, const irregular_tiled_matrix_desc_t* A,
                 const irregular_tiled_matrix_desc_t* B,
                 irregular_tiled_matrix_desc_t* C)
{
    irregular_tiled_matrix_desc_t *Cdist;
    dague_handle_t* zgemm_handle;
    dague_arena_t* arena;
    int P, Q, m, n;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_New", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_New", "illegal value of transB");
        return NULL /*-2*/;
    }
    if ( !(C->dtype & irregular_tiled_matrix_desc_type) ) {
        dplasma_error("dplasma_zgemm_New", "illegal type of descriptor for C (must be irregular_tiled_matrix_desc_t)");
        return NULL;
    }

    P = ((irregular_tiled_matrix_desc_t*)C)->grid.rows;
    Q = ((irregular_tiled_matrix_desc_t*)C)->grid.cols;

    m = dplasma_imax(C->mt, P);
    n = dplasma_imax(C->nt, Q);

    /* Create a copy of the A matrix to be used as a data distribution metric.
     * As it is used as a NULL value we must have a data_copy and a data associated
     * with it, so we can create them here.
     * Create the task distribution */
    Cdist = (irregular_tiled_matrix_desc_t*)malloc(sizeof(irregular_tiled_matrix_desc_t));

    unsigned int *itil = (unsigned int*)malloc((C->mt+1)*sizeof(unsigned int));
    unsigned int *jtil = (unsigned int*)malloc((C->nt+1)*sizeof(unsigned int));
    int k;
    for (k = 0; k < C->mt; ++k) itil[k] = C->itiling[k];
    for (k = 0; k < C->nt; ++k) jtil[k] = C->jtiling[k];

    irregular_tiled_matrix_desc_init(
        Cdist, tile_coll_RealDouble,
        C->super.nodes, C->super.myrank,
        m, n, /* Dimensions of the matrix             */
        C->mt, C->nt,
        itil, jtil,
        0, 0, /* Starting points (not important here) */
        0, 0);

    Cdist->super.super.data_of = NULL;
    Cdist->super.super.data_of_key = NULL;

    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_NN_handle_t* handle;
            handle = dague_zgemm_NN_new(transA, transB, alpha, beta,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C,
                                        (dague_ddesc_t*)Cdist);
            arena = handle->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA];
            zgemm_handle = (dague_handle_t*)handle;
        } else {
            dague_zgemm_NT_handle_t* handle;
            handle = dague_zgemm_NT_new(transA, transB, alpha, beta,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C,
                                        (dague_ddesc_t*)Cdist);
            arena = handle->arenas[DAGUE_zgemm_NT_DEFAULT_ARENA];
            zgemm_handle = (dague_handle_t*)handle;
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_TN_handle_t* handle;
            handle = dague_zgemm_TN_new(transA, transB, alpha, beta,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C,
                                        (dague_ddesc_t*)Cdist);
            arena = handle->arenas[DAGUE_zgemm_TN_DEFAULT_ARENA];
            zgemm_handle = (dague_handle_t*)handle;
        }
        else {
            dague_zgemm_TT_handle_t* handle;
            handle = dague_zgemm_TT_new(transA, transB, alpha, beta,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C,
                                        (dague_ddesc_t*)Cdist);
            arena = handle->arenas[DAGUE_zgemm_TT_DEFAULT_ARENA];
            zgemm_handle = (dague_handle_t*)handle;
        }
    }

    dplasma_add2arena_tile(arena,
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_double_complex_t, A->mb);

    return zgemm_handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zgemm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_New
 * @sa dplasma_zgemm
 *
 ******************************************************************************/
void
dplasma_zgemm_Destruct( dague_handle_t *handle )
{
    dague_zgemm_NN_handle_t *zgemm_handle = (dague_zgemm_NN_handle_t *)handle;
    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)(zgemm_handle->Cdist) );
    free( zgemm_handle->Cdist );

    dague_matrix_del2arena( ((dague_zgemm_NN_handle_t *)handle)->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA] );
    dague_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm - Performs one of the following matrix-matrix operations
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   A is not transposed;
 *          = PlasmaTrans:     A is transposed;
 *          = PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   B is not transposed;
 *          = PlasmaTrans:     B is transposed;
 *          = PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ))
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_New
 * @sa dplasma_zgemm_Destruct
 * @sa dplasma_cgemm
 * @sa dplasma_dgemm
 * @sa dplasma_sgemm
 *
 ******************************************************************************/
int
summa_zgemm( dague_context_t *dague,
             PLASMA_enum transA, PLASMA_enum transB,
             dague_complex64_t alpha, const irregular_tiled_matrix_desc_t *A,
             const irregular_tiled_matrix_desc_t *B,
             irregular_tiled_matrix_desc_t *C)
{
    dague_handle_t *dague_zgemm = NULL;
    int M, N, K;
    int Am, An, Ai, Aj, Amt, Ant;
    int Bm, Bn, Bi, Bj, Bmt, Bnt;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        dplasma_error("summa_zgemm", "illegal value of transA");
        return -1;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        dplasma_error("summa_zgemm", "illegal value of transB");
        return -2;
    }

    unsigned int *iAtiling = A->itiling;
    unsigned int *jAtiling = A->jtiling;
    unsigned int *iBtiling = B->itiling;
    unsigned int *jBtiling = B->jtiling;
    unsigned int *iCtiling = C->itiling;
    unsigned int *jCtiling = C->jtiling;

    if ( transA == PlasmaNoTrans ) {
        Am  = A->m;
        An  = A->n;
        Ai  = A->i;
        Aj  = A->j;
        Amt = A->mt;
        Ant = A->nt;
    } else {
        Am  = A->n;
        An  = A->m;
        iAtiling = A->jtiling;
        jAtiling = A->itiling;
        Ai  = A->j;
        Aj  = A->i;
        Amt = A->nt;
        Ant = A->mt;
    }

    if ( transB == PlasmaNoTrans ) {
        Bm  = B->m;
        Bn  = B->n;
        Bmb = B->mb;
        Bnb = B->nb;
        Bi  = B->i;
        Bj  = B->j;
        Bmt = B->mt;
        Bnt = B->nt;
    } else {
        Bm  = B->n;
        Bn  = B->m;
        iBtiling = B->jtiling;
        jBtiling = B->itiling;
        Bi  = B->j;
        Bj  = B->i;
        Bmt = B->nt;
        Bnt = B->mt;
    }

    int b = 1, i;
    unsigned int *iAsubtiling = iAtiling+Ai;
    unsigned int *jAsubtiling = jAtiling+Aj;
    unsigned int *iBsubtiling = iBtiling+Bi;
    unsigned int *jBsubtiling = jBtiling+Bj;
    unsigned int *iCsubtiling = iCtiling+C->i;
    unsigned int *jCsubtiling = jCtiling+C->j;

    if (Amt != Cmt || Ant != Bmt || Bnt != Cnt) {
	    dplasma_error("summa_zgemm","Symbolic tilings differ");
	    return -101;
    }

    for (i = 0; i < Amt; ++i)
	    if (iAsubtiling[i] != jCsubtiling[i])
		    b = -102;

    for (i = 0; i < Ant; ++i)
	    if (jAsubtiling[i] != iBsubtiling[i])
		    b = -103;

    for (i = 0; i < Bnt; ++i)
	    if (iAsubtiling[i] != jCsubtiling[i])
		    b = -104;

    if (b < -100) {
	    dplasma_error("summa_zgemm", "Tile sizes differ");
	    return b;
    }

    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("summa_zgemm", "sizes of submatrices have to match");
        return -101;
    }
    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("summa_zgemm", "start indexes have to match");
        return -101;
    }

    if ( !(C->dtype & irregular_tiled_matrix_desc_type) ) {
        dplasma_error("summa_zgemm", "illegal type of descriptor for C");
        return -3.;
    }

    M = C->m;
    N = C->n;
    K = An;

    /* Quick return */
    if (M == 0 || N == 0 || ((alpha == (PLASMA_Complex64_t)0.0 || K == 0))
        return 0;

    dague_zgemm = summa_zgemm_New(transA, transB,
                                  alpha, A,
                                  B,
                                  C);

    if ( dague_zgemm != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zgemm);
        dplasma_progress(dague);
        dplasma_zgemm_Destruct( dague_zgemm );
        return 0;
    }
    else {
        return -101;
    }
}
