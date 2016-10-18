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
#include "irregular_tiled_matrix.h"
#include "zsumma_NN.h"
#include "zsumma_NT.h"
#include "zsumma_TN.h"
#include "zsumma_TT.h"

typedef struct dague_function_vampire_s {
    dague_function_t super;
    dague_hook_t    *saved_prepare_input;
    void *         (*resolve_future_function)(void*);
} dague_function_vampire_t;

static int future_input_for_read_a_task(dague_execution_unit_t * context, __dague_zsumma_NN_READ_A_task_t * this_task)
{
    const dague_zsumma_NN_handle_t *__dague_handle = (dague_zsumma_NN_handle_t *) this_task->dague_handle;
    dague_function_vampire_t *vf = (dague_function_vampire_t*)this_task->function;
    dague_data_copy_t *copy = NULL;
    void *f = NULL, *tile = NULL;
    const int m = this_task->locals.m.value;
    const int k = this_task->locals.k.value;
    /** Lookup the input data, and store them in the context if any */
    assert(NULL == this_task->data.A.data_in);
    copy = dague_data_get_copy(__dague_handle->dataA->data_of(__dague_handle->dataA, m, k), 0);
    f = DAGUE_DATA_COPY_GET_PTR(copy);
    tile = vf->resolve_future_function(f);
    copy->device_private = tile;
    return vf->saved_prepare_input(context, (dague_execution_context_t *)this_task);
}

static int future_input_for_read_b_task(dague_execution_unit_t * context, __dague_zsumma_NN_READ_B_task_t * this_task)
{
    const dague_zsumma_NN_handle_t *__dague_handle = (dague_zsumma_NN_handle_t *) this_task->dague_handle;
    dague_function_vampire_t *vf = (dague_function_vampire_t*)this_task->function;
    dague_data_copy_t *copy = NULL;
    void *f = NULL, *tile = NULL;
    const int k = this_task->locals.k.value;
    const int n = this_task->locals.n.value;
    /** Lookup the input data, and store them in the context if any */
    assert(NULL == this_task->data.B.data_in);
    copy = dague_data_get_copy(__dague_handle->dataB->data_of(__dague_handle->dataB, k, n), 0);
    f = DAGUE_DATA_COPY_GET_PTR(copy);
    tile = vf->resolve_future_function(f);
    copy->device_private = tile;
    return vf->saved_prepare_input(context, (dague_execution_context_t *)this_task);
}

static int future_input_for_summa_task(dague_execution_unit_t * context, __dague_zsumma_NN_SUMMA_task_t * this_task)
{
    const dague_zsumma_NN_handle_t *__dague_handle = (dague_zsumma_NN_handle_t *) this_task->dague_handle;
    dague_function_vampire_t *vf = (dague_function_vampire_t*)this_task->function;
    dague_data_copy_t *copy = NULL;
    void *f = NULL, *tile = NULL;
    const int m = this_task->locals.m.value;
    const int n = this_task->locals.n.value;
    const int k = this_task->locals.k.value;
    if(k == 0 ) {
        /** Lookup the input data, and store them in the context if any */
        assert(NULL == this_task->data.C.data_in);
        copy = dague_data_get_copy(__dague_handle->dataC->data_of(__dague_handle->dataC, m, n), 0);
        f = DAGUE_DATA_COPY_GET_PTR(copy);
        tile = vf->resolve_future_function(f);
        copy->device_private = tile;
    }
    return vf->saved_prepare_input(context, (dague_execution_context_t *)this_task);
}

static void attach_futures_prepare_input(dague_handle_t *handle, const char *task_name, void*(*resolve_future_function)(void*))
{
    int fid;
    dague_function_vampire_t *vf;
    for(fid = 0; fid < handle->nb_functions; fid++) {
        if( strcmp(handle->functions_array[fid]->name, task_name) == 0 ) {
            break;
        }
    }
    if( fid == handle->nb_functions ) {
        fprintf(stderr, "%s:%d -- Internal Error: could not find a function with name '%s' in handle\n", __FILE__, __LINE__, task_name);
        assert(0);
        return;
    }
    assert(NULL != resolve_future_function);
    vf = (dague_function_vampire_t*)malloc(sizeof(dague_function_vampire_t));
    memcpy(&vf->super, handle->functions_array[fid], sizeof(dague_function_t));
    asprintf((char **)&vf->super.name, "%s(vampirized)", handle->functions_array[fid]->name);
    vf->saved_prepare_input = vf->super.prepare_input;
    vf->resolve_future_function = resolve_future_function;
    if( strcmp(task_name, "READ_A") == 0 )
        vf->super.prepare_input = (dague_hook_t*)future_input_for_read_a_task;
    else if( strcmp(task_name, "READ_B") == 0 )
        vf->super.prepare_input = (dague_hook_t*)future_input_for_read_b_task;
    else if( strcmp(task_name, "SUMMA") == 0 )
        vf->super.prepare_input = (dague_hook_t*)future_input_for_summa_task;
    else assert(0);
    handle->functions_array[fid] = (dague_function_t*)vf;
}

/**
 *******************************************************************************
 *
 * @ingroup summa_zsumma
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
    int i, j, k, l;

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

    irregular_tiled_matrix_desc_init(
        Cdist, tile_coll_RealDouble,
        C->super.nodes, C->super.myrank,
        m, n, /* Dimensions of the matrix             */
        C->mt, C->nt,
        C->Mtiling, C->Ntiling,
        0, 0, /* Starting points (not important here) */
        C->mt, C->nt, P, 0);

    for (i = Cdist->grid.rrank*Cdist->grid.strows; i < C->mt; i+=Cdist->grid.rows*Cdist->grid.strows)
        for (k = 0; k < Cdist->grid.stcols; ++k)
            for (j = Cdist->grid.crank*Cdist->grid.stcols; j < C->nt; j+=Cdist->grid.cols*Cdist->grid.stcols)
                for (l = 0; l < Cdist->grid.stcols; ++l) {
	                irregular_tiled_matrix_desc_set_data(Cdist, NULL, i+k, j+l, C->Mtiling[i+k], C->Ntiling[j+l], 0, ((dague_ddesc_t*)C)->rank_of((dague_ddesc_t*)C, i+k, j+l));
                }

    Cdist->super.data_of = NULL;
    Cdist->super.data_of_key = NULL;

    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            dague_zsumma_NN_handle_t* handle;
            handle = dague_zsumma_NN_new(transA, transB, alpha,
                                         (dague_ddesc_t*)A,
                                         (dague_ddesc_t*)B,
                                         (dague_ddesc_t*)C,
                                         (dague_ddesc_t*)Cdist,
                                         0);
            arena = handle->arenas[DAGUE_zsumma_NN_DEFAULT_ARENA];
            zsumma_handle = (dague_handle_t*)handle;
        } else {
            dague_zsumma_NT_handle_t* handle;
            handle = dague_zsumma_NT_new(transA, transB, alpha,
                                         (dague_ddesc_t*)A,
                                         (dague_ddesc_t*)B,
                                         (dague_ddesc_t*)C,
                                         (dague_ddesc_t*)Cdist,
                                         0);
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
                                         (dague_ddesc_t*)Cdist,
                                         0);
            arena = handle->arenas[DAGUE_zsumma_TN_DEFAULT_ARENA];
            zsumma_handle = (dague_handle_t*)handle;
        }
        else {
            dague_zsumma_TT_handle_t* handle;
            handle = dague_zsumma_TT_new(transA, transB, alpha,
                                         (dague_ddesc_t*)A,
                                         (dague_ddesc_t*)B,
                                         (dague_ddesc_t*)C,
                                         (dague_ddesc_t*)Cdist,
                                         0);
            arena = handle->arenas[DAGUE_zsumma_TT_DEFAULT_ARENA];
            zsumma_handle = (dague_handle_t*)handle;
        }
    }


    if( A->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zsumma_handle, "READ_A", A->future_resolve_fct);
    }
    if( B->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zsumma_handle, "READ_B", B->future_resolve_fct);
    }
    if( C->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zsumma_handle, "SUMMA", C->future_resolve_fct);
    }

    unsigned int max_tile = summa_imax(A->max_tile, summa_imax(B->max_tile, C->max_tile));
    unsigned int max_mb = summa_imax(A->max_mb, summa_imax(B->max_mb, C->max_mb));

    dplasma_add2arena_tile(arena,
                           /* FIXME: The size has to be optimized, the worst case will do for now */
                           max_tile*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_double_complex_t, max_mb);

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

    unsigned int *mAtiling = A->Mtiling;
    unsigned int *nAtiling = A->Ntiling;
    unsigned int *mBtiling = B->Mtiling;
    unsigned int *nBtiling = B->Ntiling;
    unsigned int *mCtiling = C->Mtiling;
    unsigned int *nCtiling = C->Ntiling;

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
        mAtiling = A->Ntiling;
        nAtiling = A->Mtiling;
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
        mBtiling = B->Ntiling;
        nBtiling = B->Mtiling;
        Bi  = B->j;
        Bj  = B->i;
        Bmt = B->nt;
        Bnt = B->mt;
    }




    int b = 1, i;
    unsigned int *mAsubtiling = mAtiling+Ai;
    unsigned int *nAsubtiling = nAtiling+Aj;
    unsigned int *mBsubtiling = mBtiling+Bi;
    unsigned int *nBsubtiling = nBtiling+Bj;
    unsigned int *mCsubtiling = mCtiling+C->i;
    unsigned int *nCsubtiling = nCtiling+C->j;

    if (Amt != C->mt || Ant != Bmt || Bnt != C->nt) {
	    dplasma_error("summa_zsumma","Symbolic tilings differ");
	    return -101;
    }

    for (i = 0; i < Amt; ++i)
	    if (mAsubtiling[i] != mCsubtiling[i])
		    b = -102;

    for (i = 0; i < Ant; ++i)
	    if (nAsubtiling[i] != mBsubtiling[i])
		    b = -103;

    for (i = 0; i < Bnt; ++i)
	    if (nBsubtiling[i] != nCsubtiling[i])
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
