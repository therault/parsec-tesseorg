/*
 * Copyright (c) 2016-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */

#include <assert.h>
#include "dplasma_bcast.h"

/*
 * Returns a key from the coordinate m, n in C
 */
parsec_key_t gemm_plan_make_key(gemm_plan_t *plan, int m, int n)
{
    return (uint64_t)n * (uint64_t)plan->mt + (uint64_t)m;
}

/*
 * n is a local column; the returned value is the local column, knowing there are s between
 * n and the returned one. Should work with s positive or negative.
 */
int gemm_plan_local_column_at_distance(gemm_plan_t *plan, int n, int s)
{
    int i, j;
    int ssign;
    ssign = (s < 0) ? -1 : 1;
    for(i = 0; plan->local_col[i] != -1; i++) {
        if( plan->local_col[i] == n ) {
            j = i;
            do {
                j += ssign;
                if( (j < 0) || (plan->local_col[j] == -1) ) {
                    return -1;
                }
            } while( (j >= 0) && (plan->local_col[j] != -1) );
            return plan->local_col[j];
        }
    }
    return -1;
}

int gemm_plan_index_of_local_column(gemm_plan_t *plan, int n)
{
    int i;
    for(i = 0; plan->local_col[i] != -1; i++)
        if( plan->local_col[i] == n )
            return i;
    return -1;
}

/*
 * Returns the highest rank that holds a B(k, n) that contributes to C(m, n)
 */
int gemm_plan_highest_rank(gemm_plan_t *plan, int m, int n)
{
    (void)m;
    (void)n;
    return plan->worldsize;
}

/*
 * Let k_i 0 <= i <= K_{m,n} be the set such that GEMM(m, n, k_i) runs on
 * rank r; This function returns k_0 
 */
int gemm_plan_kfirst(gemm_plan_t *plan, int m, int n, int r)
{
    parsec_key_t key = gemm_plan_make_key(plan, m, n);
    gemm_plan_update_list_t *l;
    if( plan->myrank != r ) return -1;
    l = parsec_hash_table_find(&plan->local_k, key);
    assert( NULL != l );
    if( l->nb == 0 )
        return -1;
    return l->updates_order[0];
}

/*
 * Let k_i 0 <= i <= K_{m,n} be the set such that GEMM(m, n, k_i) runs on
 * rank r; This function returns k_{K_{m,n}}
 */
int gemm_plan_klast(gemm_plan_t *plan, int m, int n, int r)
{
    parsec_key_t key = gemm_plan_make_key(plan, m, n);
    gemm_plan_update_list_t *l;
    if( plan->myrank != r ) return -1;
    l = parsec_hash_table_find(&plan->local_k, key);
    assert( NULL != l );
    if( l->nb == 0 )
        return -1;
    return l->updates_order[l->nb-1];
}

/*
 * Let k_i 0 <= i <= K_{m,n} be the set such that GEMM(m, n, k_i) runs on
 * rank r; Assuming k = k_i, this function returns returns k_{i+1}
 */
int gemm_plan_knext(gemm_plan_t *plan, int m, int n, int r, int k)
{
    int i;
    parsec_key_t key = gemm_plan_make_key(plan, m, n);
    gemm_plan_update_list_t *l;
    if( plan->myrank != r ) return -1;
    l = parsec_hash_table_find(&plan->local_k, key);
    assert( NULL != l );
    assert( l->nb != 0 );
    for(i = 0; i < l->nb; i++) {
        if( l->updates_order[i] == k ) {
            if( i == l->nb - 1 ) {
                return -1;
            } else {
                return l->updates_order[i+1];
            }
        }
    }
    assert( 0 /* We should have found k in the updates_order list */ );
    return -1;
}

/*
 * Let k_i 0 <= i <= K_{m,n} be the set such that GEMM(m, n, k_i) runs on
 * rank r; Assuming k = k_i, this function returns returns k_{i-1}
 */
int gemm_plan_kprev(gemm_plan_t *plan, int m, int n, int r, int k)
{
    int i;
    parsec_key_t key = gemm_plan_make_key(plan, m, n);
    gemm_plan_update_list_t *l;
    if( plan->myrank != r ) return -1;
    l = parsec_hash_table_find(&plan->local_k, key);
    assert( NULL != l );
    assert( l->nb != 0 );
    for(i = 0; i < l->nb; i++) {
        if( l->updates_order[i] == k ) {
            if( i == 0 ) {
                return -1;
            } else {
                return l->updates_order[i-1];
            }
        }
    }
    assert( 0 /* We should have found k in the updates_order list */ );
    return -1;
}

/*
 * This function returns the first rank that has local contributions
 * to C(m, n)
 */
int gemm_plan_rank_first(gemm_plan_t *plan, int m, int n)
{
    int r;
    r = plan->descC->super.rank_of(&plan->descC->super, m, n);
    return (r + 1) % plan->worldsize;
}
    
/*
 * This function returns the last rank that has local contributions
 * to C(m, n). The target should hold C(m, n).
 */
int gemm_plan_rank_last(gemm_plan_t *plan, int m, int n)
{
    int r;
    r = plan->descC->super.rank_of(&plan->descC->super, m, n);
    return r;
}

/*
 * This function returns the next rank that has local contributions
 * to C(m, n), knowing that the current contribution happened on r.
 */
int gemm_plan_rank_next(gemm_plan_t *plan, int m, int n, int r)
{
    (void)m;
    (void)n;
    return (r + 1) % plan->worldsize;
}

/*
 * This function returns the previous rank that had local contributions
 * to C(m, n), knowing that the current contribution happened on r.
 */
int gemm_plan_rank_prev(gemm_plan_t *plan, int m, int n, int r)
{
    (void)m;
    (void)n;
    return ( r + plan->worldsize - 1 ) % plan->worldsize;
}

