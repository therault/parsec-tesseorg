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
#include "dplasma_z.h"
#include "zgemm_summit_NN.h"

#include "parsec/utils/mca_param.h"
static parsec_data_collection_t TrivDist;
static int TrivDistInitialized = 0;

static parsec_data_key_t TrivDist_data_key(parsec_data_collection_t *d, ...)
{
    va_list ap;
    int r;
    (void)d;
    va_start(ap, d);
    r = va_arg(ap, int);
    va_end(ap);
    return (parsec_data_key_t)r;
}

static uint32_t TrivDist_rank_of(parsec_data_collection_t *d, ...)
{
    va_list ap;
    int r;
    (void)d;
    va_start(ap, d);
    r = va_arg(ap, int);
    va_end(ap);
    return r;
}

static uint32_t TrivDist_rank_of_key(parsec_data_collection_t *d, parsec_data_key_t key)
{
    (void)d;
    (void)key;
    return (uint32_t)key;
}

static parsec_data_t *TrivDist_data_of(parsec_data_collection_t *d, ...)
{
    (void)d;
    assert(0);
    return NULL;
}

static parsec_data_t *TrivDist_data_of_key(parsec_data_collection_t *d, parsec_data_key_t key)
{
    (void)d;
    (void)key;
    assert(0);
    return NULL;
}

static int32_t TrivDist_vpid_of(parsec_data_collection_t *d, ...)
{
    (void)d;
    return 0;
}

static int32_t TrivDist_vpid_of_key(parsec_data_collection_t *d, parsec_data_key_t key)
{
    (void)d;
    (void)key;
    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_zgemm
 *
 *  dplasma_zgemm_summit_New - Generates the taskpool that performs one of the following
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
 *          Specifies that the matrix A is transposed
 *          = PlasmaNoTrans:   A is not transposed;
 *
 * @param[in] transB
 *          Specifies that the matrix B is transposed
 *          = PlasmaNoTrans:   B is not transposed;
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
dplasma_zgemm_summit_New( PLASMA_enum transA, PLASMA_enum transB,
                          parsec_complex64_t alpha, const parsec_tiled_matrix_dc_t* A,
                          const parsec_tiled_matrix_dc_t* B,
                          parsec_tiled_matrix_dc_t* C,
                          int b, int c, int d,
                          int p, int q)
{
    parsec_taskpool_t* zgemm_handle = NULL;
    parsec_arena_t* arena;

    if( TrivDistInitialized == 0 ) {
        TrivDistInitialized = 1;
        parsec_data_collection_init(&TrivDist, A->super.nodes, A->super.myrank);
        TrivDist.data_key = TrivDist_data_key;
        TrivDist.rank_of = TrivDist_rank_of;
        TrivDist.rank_of_key = TrivDist_rank_of_key;
        TrivDist.data_of = TrivDist_data_of;
        TrivDist.data_of_key = TrivDist_data_of_key;
        TrivDist.vpid_of = TrivDist_vpid_of;
        TrivDist.vpid_of_key = TrivDist_vpid_of_key;
        TrivDist.dc_name = "TrivDist";
        TrivDist.dc_dim = "";
    }
    
    /* Check input arguments */
    if ((transA != PlasmaNoTrans)) {
        dplasma_error("zgemm_summit_New", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != PlasmaNoTrans)) {
        dplasma_error("zgemm_summit_New", "illegal value of transB");
        return NULL /*-2*/;
    }
    
    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            int u, v;
            parsec_zgemm_summit_NN_taskpool_t* handle;

            if( d*p < B->mt ) {
                fprintf(stderr, "Condition not met: d(%d) * p(%d) >= K(%d)\n",
                        d, p, B->mt);
                return NULL;
            }
            if( b*p < A->mt ) {
                fprintf(stderr, "Condition not met: b(%d) * p(%d) >= M(%d)\n",
                        b, p, A->mt);
                return NULL;
            }
            if( c*q < C->nt ) {
                fprintf(stderr, "Condition not met: c(%d) * q(%d) >= N(%d)\n",
                        c, q, C->nt);
                return NULL;
            }
            
            handle = parsec_zgemm_summit_NN_new(GEMM_SUMMIT_NN, transA, transB, alpha,
                                                A,
                                                B,
                                                C,
                                                &TrivDist,
                                                b, c, d, p, q);
            arena = handle->arenas[PARSEC_zgemm_summit_NN_DEFAULT_ARENA];

            u = B->super.myrank / q;
            v = B->super.myrank % q;

            {
                int M = A->mt;
                int Mbound = M/(p*b);
                int Mlim = p*b*Mbound + u;
                handle->_g_xMax = Mbound + (Mlim < M) - 1;
                printf("For rank (%d, %d): xMax = %d (M = %d, d=%d, MBound = %d, Mlim = %d)\n", u, v, handle->_g_xMax, M, b, Mbound, Mlim);
            }
            
            {
                int N = C->nt;
                int Nbound = N/(c*q);
                int Nlim = c*q*Nbound + v;
                handle->_g_yMax = Nbound + (Nlim < N) - 1;
                printf("For rank (%d, %d): yMax = %d (N = %d, q=%d, c=%d, NBound = %d, Nlim = %d)\n", u, v, handle->_g_yMax, N, q, c, Nbound, Nlim);
            }
            
            {
                int K = B->mt;
                int Kbound = K/(d*p);
                int Klim = d*p*Kbound + u;
                handle->_g_zMax = Kbound + (Klim < K) - 1;
                printf("For rank (%d, %d): zMax = %d (K = %d, p=%d, d=%d, KBound = %d, Klim = %d)\n", u, v, handle->_g_zMax, K, p, b, Kbound, Klim);
            }
            
            zgemm_handle = (parsec_taskpool_t*)handle;
        } 
    }

    dplasma_add2arena_tile(arena,
                           A->bsiz*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, A->mb);

    return zgemm_handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  summa_zsumma_summit_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zgemm_summit_New().
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
dplasma_zgemm_summit_Destruct( parsec_taskpool_t *tp )
{
    parsec_zgemm_summit_NN_taskpool_t *zgemm_taskpool = (parsec_zgemm_summit_NN_taskpool_t *)tp;
    if( zgemm_taskpool->_g_summa_type == GEMM_SUMMIT_NN ) {
        if (zgemm_taskpool->arenas[PARSEC_zgemm_summit_NN_DEFAULT_ARENA])
            parsec_matrix_del2arena( zgemm_taskpool->arenas[PARSEC_zgemm_summit_NN_DEFAULT_ARENA] );
    }
    parsec_taskpool_free(tp);
}

