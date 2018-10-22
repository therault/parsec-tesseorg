/*
 * Copyright (c) 2016-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */

#include <assert.h>
#include "dplasma_bcast.h"

/*
 * Returns k such that gemm_plan_red_index(plan, m, n, k) == i and
 * k is the last gemm on node i
 */
int gemm_plan_last_k_of_red_index(gemm_plan_t *plan, int m, int n, int i)
{
    assert( (m >= 0) && (m<plan->mt));
    assert( (n >= 0) && (n<plan->nt));
    assert( (i >= 0) && (i<plan->P));
    return plan->ip[(m*plan->nt+n)*plan->P + i];
}

/*
 * Returns k such that gemm_plan_red_index(plan, m, n, k) == i and
 * k is the first gemm of node i
 */
int gemm_plan_first_k_of_red_index(gemm_plan_t *plan, int m, int n, int i)
{
    assert( (m >= 0) && (m<plan->mt));
    assert( (n >= 0) && (n<plan->nt));
    assert( (i >= 0) && (i<plan->P));
    int k = plan->ip[(m*plan->nt+n)*plan->P + i];
    while( plan->prev[(m*plan->nt+n)*plan->kt + k] != -1 )
        k = plan->prev[(m*plan->nt+n)*plan->kt + k];
    return k;
}

/*
 * Returns the position in the pipeline reduction of the
 * different node contributions to C(m, n), such that
 * k is the last local contribution to C(m, n) for the calling
 * node.
 */
int gemm_plan_red_index(gemm_plan_t *plan, int m, int n, int k)
{
    int i;
    assert( (m >= 0) && (m<plan->mt));
    assert( (n >= 0) && (n<plan->nt));
    for(i = 0; i < plan->P; i++) {
        if( plan->ip[(m*plan->nt+n)*plan->P + i] == k )
            return i;
    }
    assert(0);
    return -1;
}

/*
 * Returns how many nodes contribute to C(m ,n) 
 */
int gemm_plan_max_red_index(gemm_plan_t *plan, int m, int n)
{
    int i;
    assert( (m >= 0) && (m<plan->mt));
    assert( (n >= 0) && (n<plan->nt));
    for(i = 0; i < plan->P; i++) {
        if( plan->ip[(m*plan->nt+n)*plan->P + i] == -1 )
            break;
    }
    return i-1;
}

/*
 * Returns k' such that gemm_plan_next(plan, m, n, k') = k
 * Return -1 if there is no such k'
 */
int gemm_plan_prev(gemm_plan_t *plan, int m, int n, int k)
{
    int ret;
    assert( (m >= 0) && (m<plan->mt));
    assert( (n >= 0) && (n<plan->nt));
    assert( (k >= 0) && (k<plan->kt));
    ret = plan->prev[(m*plan->nt+n)*plan->kt + k];
    return ret;
}

/*
 * GEMM(m, n, k) was a previous local contribution to C(m, n)
 * This function returns k' such that GEMM(m, n, k') is the next
 * local GEMM to execute
 * Returns -1 if there is no such k'
 */
int gemm_plan_next(gemm_plan_t *plan, int m, int n, int k)
{
    assert( (m >= 0) && (m<plan->mt));
    assert( (n >= 0) && (n<plan->nt));
    assert( (k >= 0) && (k<plan->kt));
    return plan->next[(m*plan->nt+n)*plan->kt + k];
}

