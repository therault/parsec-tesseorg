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

#include "zsumma_NN.h"
#include "zsumma_NT.h"
#include "zsumma_TN.h"
#include "zsumma_TT.h"

/**
 *******************************************************************************
 *
 * @ingroup summa_dsumma
 *
 *  summa_zsumma_New - Generates the handle that performs one of the following
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
 *          destroy with summa_zsumma_Destruct();
 *
 *******************************************************************************
 *
 * @sa summa_zsumma
 * @sa summa_zsumma_Destruct
 * @sa summa_csumma_New
 * @sa summa_dsumma_New
 * @sa summa_ssumma_New
 *
 ******************************************************************************/
dague_handle_t*
summa_zsumma_New( PLASMA_enum transA, PLASMA_enum transB,
                 dague_complex64_t alpha, const irregular_tiled_matrix_desc_t* A,
                 const irregular_tiled_matrix_desc_t* B,
                 irregular_tiled_matrix_desc_t* C)
{
    irregular_tiled_matrix_desc_t *Cdist;
    dague_handle_t* zsumma_handle;
    dague_arena_t* arena;
    int P, Q, m, n;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        dplasma_error("summa_zsumma_New", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        dplasma_error("summa_zsumma_New", "illegal value of transB");
        return NULL /*-2*/;
    }
    if ( !(C->dtype & irregular_tiled_matrix_desc_type) ) {
        dplasma_error("summa_zsumma_New", "illegal type of descriptor for C (must be irregular_tiled_matrix_desc_t)");
        return NULL;
    }

    P = ((irregular_tiled_matrix_desc_t*)C)->grid.rows;
    Q = ((irregular_tiled_matrix_desc_t*)C)->grid.cols;

    m = summa_imax(C->mt, P);
    n = summa_imax(C->nt, Q);

    /* Create a copy of the A matrix to be used as a data distribution metric.
     * As it is used as a NULL value we must have a data_copy and a data associated
     * with it, so we can create them here.
     * Create the task distribution */
    Cdist = (irregular_tiled_matrix_desc_t*)malloc(sizeof(irregular_tiled_matrix_desc_t));

    unsigned int *itil = (unsigned int*)malloc((C->lmt)*sizeof(unsigned int));
    unsigned int *jtil = (unsigned int*)malloc((C->lnt)*sizeof(unsigned int));
    int i, j, k;
    for (k = 0; k < C->lmt; ++k) itil[k] = C->itiling[k];
    for (k = 0; k < C->lnt; ++k) jtil[k] = C->jtiling[k];

    unsigned int max_tile_size = 0, max_tile_mb = 0;
    for (i = 0; i < C->lmt; ++i) {
	    if (C->jtiling[i] > max_tile_mb)
		    max_tile_mb = C->jtiling[i];
	    for (j = 0; j < C->lnt; ++j)
		    if (C->itiling[i]*C->jtiling[j] > max_tile_size)
			    /* Worst case scenario */
			    max_tile_size = C->itiling[i]*C->jtiling[j];
    }

    irregular_tiled_matrix_desc_init(
        Cdist, tile_coll_RealDouble,
        C->super.nodes, C->super.myrank,
        m, n, /* Dimensions of the matrix             */
        C->mt, C->nt,
        itil, jtil,
        0, 0, /* Starting points (not important here) */
        0, 0, P);

    Cdist->super.data_of = NULL;
    Cdist->super.data_of_key = NULL;

    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            dague_zsumma_NN_handle_t* handle;
            handle = dague_zsumma_NN_new(transA, transB, alpha,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C,
                                        (dague_ddesc_t*)Cdist);
            arena = handle->arenas[DAGUE_zsumma_NN_DEFAULT_ARENA];
            zsumma_handle = (dague_handle_t*)handle;
        } else {
            dague_zsumma_NT_handle_t* handle;
            handle = dague_zsumma_NT_new(transA, transB, alpha,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C,
                                        (dague_ddesc_t*)Cdist);
            arena = handle->arenas[DAGUE_zsumma_NT_DEFAULT_ARENA];
            zsumma_handle = (dague_handle_t*)handle;
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            dague_zsumma_TN_handle_t* handle;
            handle = dague_zsumma_TN_new(transA, transB, alpha,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C,
                                        (dague_ddesc_t*)Cdist);
            arena = handle->arenas[DAGUE_zsumma_TN_DEFAULT_ARENA];
            zsumma_handle = (dague_handle_t*)handle;
        }
        else {
            dague_zsumma_TT_handle_t* handle;
            handle = dague_zsumma_TT_new(transA, transB, alpha,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C,
                                        (dague_ddesc_t*)Cdist);
            arena = handle->arenas[DAGUE_zsumma_TT_DEFAULT_ARENA];
            zsumma_handle = (dague_handle_t*)handle;
        }
    }

    dplasma_add2arena_tile(arena,
                           /* FIXME: The size has to be optimized, the worst case will do for now */
                           max_tile_size*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_double_complex_t, max_tile_mb);

    return zsumma_handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  summa_zsumma_Destruct - Free the data structure associated to an handle
 *  created with summa_zsumma_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa summa_zsumma_New
 * @sa summa_zsumma
 *
 ******************************************************************************/
void
summa_zsumma_Destruct( dague_handle_t *handle )
{
    dague_zsumma_NN_handle_t *zsumma_handle = (dague_zsumma_NN_handle_t *)handle;
    irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)(zsumma_handle->Cdist) );
    free( zsumma_handle->Cdist );

    dague_matrix_del2arena( ((dague_zsumma_NN_handle_t *)handle)->arenas[DAGUE_zsumma_NN_DEFAULT_ARENA] );
    dague_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  summa_zsumma - Performs one of the following matrix-matrix operations
 *
 *    \f[ C = \alpha [op( A )\times op( B )],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha is scalar, and A, B and C  are matrices, with op( A )
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
 *
 * @param[out] C
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
 * @sa summa_zsumma_New
 * @sa summa_zsumma_Destruct
 * @sa summa_csumma
 * @sa summa_dsumma
 * @sa summa_ssumma
 *
 ******************************************************************************/
int
summa_zsumma( dague_context_t *dague,
             PLASMA_enum transA, PLASMA_enum transB,
             dague_complex64_t alpha, const irregular_tiled_matrix_desc_t *A,
             const irregular_tiled_matrix_desc_t *B,
             irregular_tiled_matrix_desc_t *C)
{
    dague_handle_t *dague_zsumma = NULL;
    int M, N, K;
    int Am, An, Ai, Aj, Amt, Ant;
    int Bm, Bn, Bi, Bj, Bmt, Bnt;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        dplasma_error("summa_zsumma", "illegal value of transA");
        return -1;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        dplasma_error("summa_zsumma", "illegal value of transB");
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

    if (Amt != C->mt || Ant != Bmt || Bnt != C->nt) {
	    dplasma_error("summa_zsumma","Symbolic tilings differ");
	    return -101;
    }

    for (i = 0; i < Amt; ++i)
	    if (iAsubtiling[i] != iCsubtiling[i])
		    b = -102;

    for (i = 0; i < Ant; ++i)
	    if (jAsubtiling[i] != iBsubtiling[i])
		    b = -103;

    for (i = 0; i < Bnt; ++i)
	    if (jBsubtiling[i] != jCsubtiling[i])
		    b = -104;

    if (b < -100) {
	    dplasma_error("summa_zsumma", "Tile sizes differ");
	    return b;
    }

    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("summa_zsumma", "sizes of submatrices have to match");
        return -101;
    }

    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("summa_zsumma", "start indexes have to match");
        return -101;
    }

    if ( !(C->dtype & irregular_tiled_matrix_desc_type) ) {
        dplasma_error("summa_zsumma", "illegal type of descriptor for C");
        return -3.;
    }

    M = C->m;
    N = C->n;
    K = An;

    /* Quick return */
    if (M == 0 || N == 0 || ((alpha == (PLASMA_Complex64_t)0.0 || K == 0)))
        return 0;

    dague_zsumma = summa_zsumma_New(transA, transB,
                                  alpha, A,
                                  B,
                                  C);

    if ( dague_zsumma != NULL ) {
        dague_enqueue( dague, (dague_handle_t*)dague_zsumma);
        dplasma_progress(dague);
        summa_zsumma_Destruct( dague_zsumma );
        return 0;
    }
    else {
        return -101;
    }
}
