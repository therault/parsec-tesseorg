/*
 * Copyright (c) 2016      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include <string.h>

#include "dplasma.h"
#include "dplasma_bcast.h"
#include "dplasma/include/dplasmatypes.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/irregular_tiled_matrix.h"
#include "dplasma_z.h"
#include "zsumma_NN.h"
#include "zsumma_NT.h"
#include "zsumma_TN.h"
#include "zsumma_TT.h"
#include "zgemm_bcast_NN.h"

#include "parsec/utils/mca_param.h"

typedef struct parsec_tc_vampire_s {
    parsec_task_class_t    super;
    parsec_hook_t         *saved_prepare_input;
    parsec_destruct_fn_t   saved_destructor;
    parsec_task_class_t   *saved_tc;
    void *         (*resolve_future_function)(void*, void*, void*);
} parsec_tc_vampire_t;

static int future_input_for_read_a_task(parsec_execution_stream_t * es, __parsec_zsumma_NN_READ_A_task_t * this_task)
{
    const parsec_zsumma_NN_taskpool_t *__parsec_tp = (parsec_zsumma_NN_taskpool_t *) this_task->taskpool;
    parsec_tc_vampire_t *vf = (parsec_tc_vampire_t*)this_task->task_class;
    parsec_data_copy_t *copy = NULL;
    void *f = NULL, *tile = NULL;
    const int m = this_task->locals.m.value;
    const int k = this_task->locals.k.value;
    /** Lookup the input data, and store them in the es if any */
    assert(NULL == this_task->data._f_A.data_in);
    copy = parsec_data_get_copy(((parsec_data_collection_t*)__parsec_tp->_g_descA)->data_of(((parsec_data_collection_t*)__parsec_tp->_g_descA), m, k), 0);
    f = PARSEC_DATA_COPY_GET_PTR(copy);
    tile = vf->resolve_future_function(f, es, this_task);
    if( NULL != tile ) {
        copy->device_private = tile;
        return vf->saved_prepare_input(es, (parsec_task_t *)this_task);
    } else {
        return PARSEC_HOOK_RETURN_ASYNC;
    }
}

static int future_input_for_read_b_task(parsec_execution_stream_t * es, __parsec_zsumma_NN_READ_B_task_t * this_task)
{
    const parsec_zsumma_NN_taskpool_t *__parsec_tp = (parsec_zsumma_NN_taskpool_t *) this_task->taskpool;
    parsec_tc_vampire_t *vf = (parsec_tc_vampire_t*)this_task->task_class;
    parsec_data_copy_t *copy = NULL;
    void *f = NULL, *tile = NULL;
    const int k = this_task->locals.k.value;
    const int n = this_task->locals.n.value;
    /** Lookup the input data, and store them in the es if any */
    assert(NULL == this_task->data._f_B.data_in);
    copy = parsec_data_get_copy(((parsec_data_collection_t*)__parsec_tp->_g_descB)->data_of(((parsec_data_collection_t*)__parsec_tp->_g_descB), k, n), 0);
    f = PARSEC_DATA_COPY_GET_PTR(copy);
    tile = vf->resolve_future_function(f, es, this_task);
    if( NULL != tile ) {
        copy->device_private = tile;
        return vf->saved_prepare_input(es, (parsec_task_t *)this_task);
    } else {
        return PARSEC_HOOK_RETURN_ASYNC;
    }
}

static int future_input_for_summa_task(parsec_execution_stream_t * es, __parsec_zsumma_NN_SUMMA_task_t * this_task)
{
    const parsec_zsumma_NN_taskpool_t *__parsec_tp = (parsec_zsumma_NN_taskpool_t *) this_task->taskpool;
    parsec_tc_vampire_t *vf = (parsec_tc_vampire_t*)this_task->task_class;
    parsec_data_copy_t *copy = NULL;
    void *f = NULL, *tile = NULL;
    const int m = this_task->locals.m.value;
    const int n = this_task->locals.n.value;
    const int k = this_task->locals.k.value;
    if(k == 0 ) {
        /** Lookup the input data, and store them in the es if any */
        assert(NULL == this_task->data._f_C.data_in);
        copy = parsec_data_get_copy(((parsec_data_collection_t*)__parsec_tp->_g_descC)->data_of(((parsec_data_collection_t*)__parsec_tp->_g_descC), m, n), 0);
        f = PARSEC_DATA_COPY_GET_PTR(copy);
        tile = vf->resolve_future_function(f, es, this_task);
        if( NULL != tile ) {
            copy->device_private = tile;
        } else {
            return PARSEC_HOOK_RETURN_ASYNC;
        }
    }
    return vf->saved_prepare_input(es, (parsec_task_t *)this_task);
}

static int future_input_for_accumulate_c_task(parsec_execution_stream_t * es, __parsec_zgemm_bcast_NN_ACCUMULATE_C_task_t * this_task)
{
    const parsec_zgemm_bcast_NN_taskpool_t *__parsec_tp = (parsec_zgemm_bcast_NN_taskpool_t *) this_task->taskpool;
    parsec_tc_vampire_t *vf = (parsec_tc_vampire_t*)this_task->task_class;
    parsec_data_copy_t *copy = NULL;
    void *f = NULL, *tile = NULL;
    const int m = this_task->locals.m.value;
    const int n = this_task->locals.n.value;
    const int i = this_task->locals.i.value;
    if( i == 0 ) {
        /** Lookup the input data, and store them in the es if any */
        assert(NULL == this_task->data._f_C.data_in);
        copy = parsec_data_get_copy(((parsec_data_collection_t*)__parsec_tp->_g_descC)->data_of(((parsec_data_collection_t*)__parsec_tp->_g_descC), m, n), 0);
        f = PARSEC_DATA_COPY_GET_PTR(copy);
        tile = vf->resolve_future_function(f, es, this_task);
        if( NULL != tile ) {
            copy->device_private = tile;
            return vf->saved_prepare_input(es, (parsec_task_t *)this_task);
        } else {
            return PARSEC_HOOK_RETURN_ASYNC;
        }
    }
    return vf->saved_prepare_input(es, (parsec_task_t *)this_task);
}

static void vtp_destructor(parsec_taskpool_t *tp)
{
    parsec_task_class_t *rf;
    parsec_destruct_fn_t destructor = NULL;
    parsec_tc_vampire_t *vf;
    unsigned int fid;
    for(fid = 0; fid < tp->nb_task_classes; fid++) {
        if( strstr(tp->task_classes_array[fid]->name, "(vampirized)") ) {
            vf = (parsec_tc_vampire_t*)tp->task_classes_array[fid];
            rf = vf->saved_tc;
            if(NULL == destructor) /* Do it once, or you'll call vtp_destructor again */
                destructor = vf->saved_destructor;
            free((char*)tp->task_classes_array[fid]->name);
            free((void*)tp->task_classes_array[fid]);
            tp->task_classes_array[fid] = rf; 
        }
    }
    if( NULL != destructor ) {
        fprintf(stderr, "Calling destructor\n");
        destructor(tp);
    }
}

static void attach_futures_prepare_input(parsec_taskpool_t *tp, const char *task_name, void*(*resolve_future_function)(void*, void*, void*))
{
    unsigned int fid;
    parsec_tc_vampire_t *vf;
    for(fid = 0; fid < tp->nb_task_classes; fid++) {
        if( strcmp(tp->task_classes_array[fid]->name, task_name) == 0 ) {
            break;
        }
    }
    if( fid == tp->nb_task_classes ) {
        fprintf(stderr, "%s:%d -- Internal Error: could not find a task class with name '%s' in the taskpool\n", __FILE__, __LINE__, task_name);
        assert(0);
        return;
    }
    assert(NULL != resolve_future_function);
    vf = (parsec_tc_vampire_t*)malloc(sizeof(parsec_tc_vampire_t));
    vf->saved_tc = (parsec_task_class_t*)tp->task_classes_array[fid];
    memcpy(&vf->super, tp->task_classes_array[fid], sizeof(parsec_task_class_t));
    asprintf((char **)&vf->super.name, "%s(vampirized)", tp->task_classes_array[fid]->name);
    vf->saved_prepare_input = vf->super.prepare_input;
    vf->saved_destructor = tp->destructor;
    vf->resolve_future_function = resolve_future_function;
    if( strcmp(task_name, "READ_A") == 0 )
        vf->super.prepare_input = (parsec_hook_t*)future_input_for_read_a_task;
    else if( strcmp(task_name, "READ_B") == 0 )
        vf->super.prepare_input = (parsec_hook_t*)future_input_for_read_b_task;
    else if( strcmp(task_name, "SUMMA") == 0 )
        vf->super.prepare_input = (parsec_hook_t*)future_input_for_summa_task;
    else if( strcmp(task_name, "ACCUMULATE_C") == 0 )
        vf->super.prepare_input = (parsec_hook_t*)future_input_for_accumulate_c_task;
    else exit(3);
    tp->destructor = vtp_destructor;
    tp->task_classes_array[fid] = (parsec_task_class_t*)vf;
}

static int
zsumma_check_operation_valid(PLASMA_enum transA, PLASMA_enum transB,
                             parsec_complex64_t alpha,
                             const irregular_tiled_matrix_desc_t *A,
                             const irregular_tiled_matrix_desc_t *B,
                             irregular_tiled_matrix_desc_t *C)
{
    (void)alpha;
    int b = 1, i;
    unsigned int *mAtiling = A->Mtiling;
    unsigned int *nAtiling = A->Ntiling;
    unsigned int *mBtiling = B->Mtiling;
    unsigned int *nBtiling = B->Ntiling;
    unsigned int *mCtiling = C->Mtiling;
    unsigned int *nCtiling = C->Ntiling;

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

    return b;
}

static float dplasma_zgemm_NN_bcast_cuda_cutoff_ratio = 0.0;

static parsec_hook_return_t dplasma_zgemm_NN_bcast_cuda_cutoff(const parsec_task_t *task)
{
    const __parsec_zgemm_bcast_NN_GEMM_task_t * this_task = (const __parsec_zgemm_bcast_NN_GEMM_task_t *)task;
    const parsec_zgemm_bcast_NN_taskpool_t *tp = (const parsec_zgemm_bcast_NN_taskpool_t*)this_task->taskpool;
    int m, n, k;
    int a_mb, a_nb, b_mb, b_nb, c_mb, c_nb;
    float mem;
    float flops;
    float ratio;
    
    m = this_task->locals.m.value;
    n = this_task->locals.n.value;
    k = this_task->locals.k.value;

    a_mb = tp->_g_descA->Mtiling[m];
    a_nb = tp->_g_descA->Ntiling[k];
    b_mb = tp->_g_descB->Mtiling[k];
    b_nb = tp->_g_descB->Ntiling[n];
    c_mb = tp->_g_descC->Mtiling[m];
    c_nb = tp->_g_descC->Ntiling[n];

    mem = a_mb * a_nb + b_mb * b_nb + c_mb * c_nb;
    flops = 2.0 * a_mb * a_nb * c_nb;

    ratio = flops/mem;

    if( ratio < dplasma_zgemm_NN_bcast_cuda_cutoff_ratio ) {
        return PARSEC_HOOK_RETURN_NEXT;
    }
    return PARSEC_HOOK_RETURN_DONE;
}

parsec_taskpool_t*
dplasma_zgemm_bcast_New( PLASMA_enum transA, PLASMA_enum transB,
                         parsec_complex64_t alpha, const irregular_tiled_matrix_desc_t* A,
                         const irregular_tiled_matrix_desc_t* B,
                         irregular_tiled_matrix_desc_t* C)
{
    parsec_taskpool_t* zgemm_handle;
    parsec_arena_t* arena;
    int P, Q, m, n, i, j, k, rank, nb;
    gemm_plan_t *plan;
    static char *cutoff_str = NULL;
    int *dev_index = NULL;

    if( NULL == cutoff_str ) {
        parsec_mca_param_reg_string_name("dplasma", "zgemm_bcast_cutoff_ratio",
                                         "CUDA Cutoff ratio: any task with arithmetic intensity lower than this will never be scheduled on a GPU",
                                         false, false, "0.0", &cutoff_str);
        dplasma_zgemm_NN_bcast_cuda_cutoff_ratio = strtof(cutoff_str, NULL);
        parsec_debug_verbose(0, parsec_debug_output, "Cutoff ratio set to %f for zgemm_bcast", dplasma_zgemm_NN_bcast_cuda_cutoff_ratio);
    }
    
    /* Check input arguments */
    if ((transA != PlasmaNoTrans)) {
        dplasma_error("summa_zgemm_bcast_New", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != PlasmaNoTrans)) {
        dplasma_error("summa_zgemm_bcast_New", "illegal value of transB");
        return NULL /*-2*/;
    }
    if ( !(C->dtype & irregular_tiled_matrix_desc_type) ) {
        dplasma_error("summa_zgemm_bcast_New", "illegal type of descriptor for C (must be irregular_tiled_matrix_desc_t)");
        return NULL;
    }

    P = ((irregular_tiled_matrix_desc_t*)C)->grid.rows;
    Q = ((irregular_tiled_matrix_desc_t*)C)->grid.cols;
    plan = (gemm_plan_t*)malloc(sizeof(gemm_plan_t));
    plan->P = P*Q;
    plan->mt = C->mt;
    plan->nt = C->nt;
    plan->kt = B->mt;
    plan->ip   = malloc(C->mt * C->nt * plan->P * sizeof(int));
    plan->prev = malloc(C->mt * C->nt * B->mt * sizeof(int));
    plan->next = malloc(C->mt * C->nt * B->mt * sizeof(int));
    int *lastk = malloc(plan->P * sizeof(int));
    for(m = 0; m < C->mt; m++) {
        for(n = 0; n < C->nt; n++) {
            for(i = 0; i < plan->P; i++)
                lastk[i] = -1;
            for(k = 0; k < B->mt; k++) {
                /* Cubic loop to determine, for each C(m, n),
                 * what are the local GEMM segments */
                rank = B->super.rank_of((parsec_data_collection_t*)B, k, n);
                plan->prev[(m*plan->nt+n)*plan->kt + k] = lastk[rank];
                if( -1 != lastk[rank] )
                    plan->next[(m*plan->nt+n)*plan->kt + lastk[rank]] = k;
                lastk[rank] = k;
            }
            /* Mark the last ones as finals */
            for(i = 0; i < plan->P; i++)
                if( -1 != lastk[i] )
                    plan->next[(m*plan->nt+n)*plan->kt + lastk[i]] = -1;
            /* Now, compute the reduction indexes:
             *  - Start with rank next to the host of C(m, n), so we can end on C(m, n)
             *    This is used in an attempt to distribute the order of reductions
             *  - Remember the last k used by each rank in the index/process array
             */
            rank = C->super.rank_of((parsec_data_collection_t*)C, m, n);
            j = 0;
            i = rank;
            do {
                i = (i+1)%plan->P;
                if( lastk[i] != -1 ) {
                    plan->ip[(m*plan->nt+n)*plan->P + j] = lastk[i];
                    j++;
                }
            } while(i != rank);
            assert(j != 0);
            for(; j < plan->P; j++)
                plan->ip[(m*plan->nt+n)*plan->P + j] = -1;
        }
    }

#if 0
    if( A->super.myrank == 0 ) {
        printf("Distribution of C:\n");
        for(m = 0; m < C->mt; m++) {
            printf(" ");
            for(n = 0; n < C->nt; n++) {
                rank = C->super.rank_of((parsec_data_collection_t*)C, m, n);
                printf("%d", rank);
            }
            printf("\n");
        }
        printf("Distribution of A:\n");
        for(m = 0; m < A->mt; m++) {
            printf(" ");
            for(n = 0; n < A->nt; n++) {
                rank = A->super.rank_of((parsec_data_collection_t*)A, m, n);
                printf("%d", rank);
            }
            printf("\n");
        }
        printf("Distribution of B:\n");
        for(m = 0; m < B->mt; m++) {
            printf(" ");
            for(n = 0; n < B->nt; n++) {
                rank = B->super.rank_of((parsec_data_collection_t*)B, m, n);
                printf("%d", rank);
            }
            printf("\n");
        }
        for(m = 0; m < C->mt; m++) {
            for(n = 0; n < C->nt; n++) {
                for(int kk = 0; kk < B->mt; kk++) {
                    if( plan->prev[(m*plan->nt+n)*plan->kt + kk] == -1 ) {
                        k = kk;
                        do {
                            rank = B->super.rank_of((parsec_data_collection_t*)B, k, n);
                            printf("G(%d,%d,%d)on(%d),", m, n, k, rank);
                            k = plan->next[(m*plan->nt+n)*plan->kt + k];
                        } while(k != -1);
                        printf("\n");
                    }
                }
            }
        }
    }
#endif
    free(lastk);

    nb = 0;
    for(i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_t *dev = parsec_devices_get(i);
        if( PARSEC_DEV_CUDA == dev->type ) {
            nb++;
        }
    }
    dev_index = (int*)malloc(nb * sizeof(int));
    nb = 0;
    for(i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_t *dev = parsec_devices_get(i);
        if( PARSEC_DEV_CUDA == dev->type ) {
            dev_index[nb++] = dev->device_index;
        }
    }
    
    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            parsec_zgemm_bcast_NN_taskpool_t* handle;
            parsec_task_class_t *gemm_tc;
            
            handle = parsec_zgemm_bcast_NN_new(GEMM_BCAST_NN, transA, transB, alpha,
                                               (const irregular_tiled_matrix_desc_t *)A,
                                               (const irregular_tiled_matrix_desc_t *)B,
                                               (irregular_tiled_matrix_desc_t *)C,
                                               (parsec_data_collection_t*)B,
                                               plan,
                                               nb, dev_index);
            arena = handle->arenas[PARSEC_zgemm_bcast_NN_DEFAULT_ARENA];

            assert( 0 == strcmp(handle->super.task_classes_array[4]->name, "GEMM") );
            gemm_tc = (parsec_task_class_t*)handle->super.task_classes_array[4];
            for(i = 0; ; i++) {
                if( PARSEC_DEV_NONE == gemm_tc->incarnations[i].type )
                    break;
                if( PARSEC_DEV_CUDA == gemm_tc->incarnations[i].type ) {
                    ((__parsec_chore_t *)&gemm_tc->incarnations[i])->evaluate = dplasma_zgemm_NN_bcast_cuda_cutoff;
                    break;
                }
            }

            handle->_g_summa_type = GEMM_BCAST_NN;
            
            zgemm_handle = (parsec_taskpool_t*)handle;
        } 
    }

    if( A->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zgemm_handle, "READ_A", A->future_resolve_fct);
    }
    if( B->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zgemm_handle, "READ_B", B->future_resolve_fct);
    }
    if( C->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zgemm_handle, "ACCUMULATE_C", C->future_resolve_fct);
    }

    parsec_datatype_t mtype;
    parsec_type_create_contiguous(1, parsec_datatype_double_complex_t, &mtype);

    parsec_arena_construct(arena, sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           mtype);

    return zgemm_handle;
}


/**
 *******************************************************************************
 *
 * @ingroup summa_zsumma
 *
 *  summa_zsumma_New - Generates the taskpool that performs one of the following
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
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with summa_zsumma_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zsumma
 * @sa dplasma_zsumma_Destruct
 * @sa dplasma_csumma_New
 * @sa dplasma_dsumma_New
 * @sa dplasma_ssumma_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zsumma_New( PLASMA_enum transA, PLASMA_enum transB,
                    parsec_complex64_t alpha, const irregular_tiled_matrix_desc_t* A,
                    const irregular_tiled_matrix_desc_t* B,
                    irregular_tiled_matrix_desc_t* C)
{
    two_dim_block_cyclic_t *Cdist;
    parsec_taskpool_t* zsumma_taskpool;
    parsec_arena_t* arena;
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

    if( ((transA == PlasmaNoTrans) && (transB == PlasmaNoTrans)) ) {
        return dplasma_zgemm_bcast_New(transA, transB, alpha, A, B, C);
    }

    P = ((irregular_tiled_matrix_desc_t*)C)->grid.rows;
    Q = ((irregular_tiled_matrix_desc_t*)C)->grid.cols;

    m = (C->mt > P) ? C->mt : P;
    n = (C->nt > Q) ? C->nt : Q;

    Cdist = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

        two_dim_block_cyclic_init(
            Cdist, matrix_RealDouble, matrix_Tile,
            C->super.nodes, C->super.myrank,
            1, 1, /* Dimensions of the tiles              */
            m, n, /* Dimensions of the matrix             */
            0, 0, /* Starting points (not important here) */
            m, n, /* Dimensions of the submatrix          */
            1, 1, P);
        Cdist->super.super.data_of = NULL;
        Cdist->super.super.data_of_key = NULL;

    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            parsec_zsumma_NN_taskpool_t* tp;
            tp = parsec_zsumma_NN_new(SUMMA_NN, transA, transB, alpha,
                                          (const irregular_tiled_matrix_desc_t *)A,
                                          (const irregular_tiled_matrix_desc_t *)B,
                                          (irregular_tiled_matrix_desc_t *)C,
                                          Cdist,
                                          0/*createC*/);
            arena = tp->arenas[PARSEC_zsumma_NN_DEFAULT_ARENA];
            zsumma_taskpool = (parsec_taskpool_t*)tp;
        } else {
            parsec_zsumma_NT_taskpool_t* tp;
            tp = parsec_zsumma_NT_new(SUMMA_NT, transA, transB, alpha,
                                          (const irregular_tiled_matrix_desc_t *)A,
                                          (const irregular_tiled_matrix_desc_t *)B,
                                          (irregular_tiled_matrix_desc_t *)C,
                                          Cdist,
                                          0);
            arena = tp->arenas[PARSEC_zsumma_NT_DEFAULT_ARENA];
            zsumma_taskpool = (parsec_taskpool_t*)tp;
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            parsec_zsumma_TN_taskpool_t* tp;
            tp = parsec_zsumma_TN_new(SUMMA_TN, transA, transB, alpha,
                                          (const irregular_tiled_matrix_desc_t *)A,
                                          (const irregular_tiled_matrix_desc_t *)B,
                                          (irregular_tiled_matrix_desc_t *)C,
                                          Cdist,
                                          0);
            arena = tp->arenas[PARSEC_zsumma_TN_DEFAULT_ARENA];
            zsumma_taskpool = (parsec_taskpool_t*)tp;
        }
        else {
            parsec_zsumma_TT_taskpool_t* tp;
            tp = parsec_zsumma_TT_new(SUMMA_TT, transA, transB, alpha,
                                          (const irregular_tiled_matrix_desc_t *)A,
                                          (const irregular_tiled_matrix_desc_t *)B,
                                          (irregular_tiled_matrix_desc_t *)C,
                                          Cdist,
                                          0);
            arena = tp->arenas[PARSEC_zsumma_TT_DEFAULT_ARENA];
            zsumma_taskpool = (parsec_taskpool_t*)tp;
        }
    }


    if( A->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zsumma_taskpool, "READ_A", A->future_resolve_fct);
    }
    if( B->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zsumma_taskpool, "READ_B", B->future_resolve_fct);
    }
    if( C->future_resolve_fct != NULL ) {
        attach_futures_prepare_input(zsumma_taskpool, "SUMMA", C->future_resolve_fct);
    }

    parsec_datatype_t mtype;
    parsec_type_create_contiguous(1, parsec_datatype_double_complex_t, &mtype);

    parsec_arena_construct(arena, sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           mtype);

    return zsumma_taskpool;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  summa_zsumma_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zsumma_New().
 *
 *******************************************************************************
 *
 * @param[in,out] tp
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsumma_New
 * @sa dplasma_zsumma
 *
 ******************************************************************************/
void
dplasma_zsumma_Destruct( parsec_taskpool_t *tp )
{
    parsec_zsumma_NN_taskpool_t *zsumma_taskpool = (parsec_zsumma_NN_taskpool_t *)tp;
    if( zsumma_taskpool->_g_summa_type == SUMMA_NN ||
        zsumma_taskpool->_g_summa_type == SUMMA_NT ||
        zsumma_taskpool->_g_summa_type == SUMMA_TN ||
        zsumma_taskpool->_g_summa_type == SUMMA_TT ) {
        
        if ( zsumma_taskpool->_g_Cdist != NULL ) {
            parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)(zsumma_taskpool->_g_Cdist) );
            free( (parsec_tiled_matrix_dc_t*)zsumma_taskpool->_g_Cdist );
            zsumma_taskpool->_g_Cdist = NULL;
        }
        if (zsumma_taskpool->arenas[PARSEC_zsumma_NN_DEFAULT_ARENA])
            parsec_matrix_del2arena( zsumma_taskpool->arenas[PARSEC_zsumma_NN_DEFAULT_ARENA] );
    }
    if( zsumma_taskpool->_g_summa_type == GEMM_BCAST_NN ) {
        parsec_zgemm_bcast_NN_taskpool_t *zgemm_bcast_NN_tp = (parsec_zgemm_bcast_NN_taskpool_t *)tp;
        if (zgemm_bcast_NN_tp->arenas[PARSEC_zgemm_bcast_NN_DEFAULT_ARENA])
            parsec_matrix_del2arena( zgemm_bcast_NN_tp->arenas[PARSEC_zgemm_bcast_NN_DEFAULT_ARENA] );
        free(zgemm_bcast_NN_tp->_g_plan->ip);
        free(zgemm_bcast_NN_tp->_g_plan->prev);
        free(zgemm_bcast_NN_tp->_g_plan->next);
        free(zgemm_bcast_NN_tp->_g_plan);
    }
    parsec_taskpool_free(tp);
}

void
dplasma_zsumma_recursive_Destruct(parsec_taskpool_t *tp)
{
    parsec_zsumma_NN_taskpool_t *zsumma_taskpool = (parsec_zsumma_NN_taskpool_t *)tp;
    if ( zsumma_taskpool->_g_Cdist != NULL ) {
        parsec_tiled_matrix_dc_destroy( (parsec_tiled_matrix_dc_t*)(zsumma_taskpool->_g_Cdist) );
        free( (parsec_tiled_matrix_dc_t*)zsumma_taskpool->_g_Cdist );
    }
    parsec_taskpool_free(tp);
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
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
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
 * @sa dplasma_zsumma_New
 * @sa dplasma_zsumma_Destruct
 * @sa dplasma_csumma
 * @sa dplasma_dsumma
 * @sa dplasma_ssumma
 *
 ******************************************************************************/
int
dplasma_zsumma(parsec_context_t *parsec,
               PLASMA_enum transA, PLASMA_enum transB,
               parsec_complex64_t alpha, const irregular_tiled_matrix_desc_t *A,
               const irregular_tiled_matrix_desc_t *B,
               irregular_tiled_matrix_desc_t *C)
{
    parsec_taskpool_t *parsec_zsumma = NULL;
    int M, N, K;

    zsumma_check_operation_valid(transA, transB, alpha, A, B, C);

    M = C->m;
    N = C->n;
    K = (transA == PlasmaNoTrans) ? A->n : A->m;

    /* Quick return */
    if (M == 0 || N == 0 || ((alpha == (PLASMA_Complex64_t)0.0 || K == 0)))
        return 0;

    parsec_zsumma = dplasma_zsumma_New(transA, transB,
                                     alpha, A,
                                     B,
                                     C);

    if ( parsec_zsumma != NULL ) {
        parsec_enqueue( parsec, (parsec_taskpool_t*)parsec_zsumma);
        dplasma_wait_until_completion(parsec);
        dplasma_zsumma_Destruct( parsec_zsumma );
        return 0;
    }
    else {
        return -101;
    }
}

#if defined(PARSEC_HAVE_RECURSIVE)
void
dplasma_zsumma_setrecursive(parsec_taskpool_t *tp, int bigtile, int opttile)
{
    parsec_zsumma_NN_taskpool_t *parsec_zsumma = (parsec_zsumma_NN_taskpool_t*)tp;
    if (bigtile > 0 && opttile > 0) {
        parsec_zsumma->_g_bigtile = bigtile;
        parsec_zsumma->_g_opttile = opttile;
    }
}


int
dplasma_zsumma_rec(parsec_context_t *parsec,
                 PLASMA_enum transA, PLASMA_enum transB,
                 parsec_complex64_t alpha,
                 const irregular_tiled_matrix_desc_t *A,
                 const irregular_tiled_matrix_desc_t *B,
                 irregular_tiled_matrix_desc_t *C, int bigtile, int opttile)
{
    parsec_taskpool_t *parsec_zsumma = NULL;

    zsumma_check_operation_valid(transA, transB, alpha, A, B, C);

    parsec_zsumma = dplasma_zsumma_New(transA, transB, alpha, A, B, C);

    if (parsec_zsumma) {
        parsec_enqueue(parsec, parsec_zsumma);
        dplasma_zsumma_setrecursive(parsec_zsumma, bigtile, opttile);
        dplasma_progress(parsec);
        dplasma_zsumma_recursive_Destruct(parsec_zsumma);
        parsec_taskpool_sync_ids();
    }

    return 0;
}
#endif
