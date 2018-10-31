/*
 * Copyright (c) 2016-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _DPLASMA_BCAST_H_
#define _DPLASMA_BCAST_H_

#include "parsec.h"
#include "dplasma.h"

BEGIN_C_DECLS

/* Helper structures and functions */

struct gemm_plan_s {
    int mt;
    int nt;
    int kt;
    int P;
    int *prev;  /* prev[m,n,k]: is the previous local GEMM to contribute to C(m ,n) */
    int *next;  /* next[m,n,k]: next GEMM to do after GEMM on k for C(m,n) */
    int *ip;    /* ip[m, n, x], 0 <= x <= P, 
                 * defines the reduction index / last GEMM contribution pair.
                 * if ip[m, n, x] == -1, then there are x-1 elements in this array
                 * There cannot be more than P elements in this array.
                 * Otherwise, ip[m, n, x] is the k such that GEMM(m, n, k) was
                 * the last contribution of a node, and that contribution is at
                 * index x in the reduction pipeline.
                 */
};

/* This will define a GEMM execution plan for the bcast_gemm interface */
typedef struct gemm_plan_s gemm_plan_t;

/*
 * Returns k such that gemm_plan_red_index(plan, m, n, k) == i and GEMM(m, n, k)
 * is the last GEMM applied on rank i
 */
int gemm_plan_last_k_of_red_index(gemm_plan_t *plan, int m, int n, int i);

/*
 * Returns k such that gemm_plan_red_index(plan, m, n, k) == i and GEMM(m, n, k)
 * is the first GEMM applied on rank i
 */
int gemm_plan_first_k_of_red_index(gemm_plan_t *plan, int m, int n, int i);

/*
 * Returns i such that gemm_plan_first_k_of_red_index(plan, m, n, i) == k and
 * k is the last gemm of node i
 */
int gemm_plan_i_for_k(gemm_plan_t *plan, int m, int n, int k);

/*
 * Returns the position in the pipeline reduction of the
 * different node contributions to C(m, n), such that
 * k is the last local contribution to C(m, n) for the calling
 * node.
 */
int gemm_plan_red_index(gemm_plan_t *plan, int m, int n, int k);
/*
 * Returns how many nodes contribute to C(m ,n) 
 */
int gemm_plan_max_red_index(gemm_plan_t *plan, int m, int n);
/*
 * Returns k' such that gemm_plan_next(plan, m, n, k') = k
 * Return -1 if there is no such k'
 */
int gemm_plan_prev(gemm_plan_t *plan, int m, int n, int k);
/*
 * GEMM(m, n, k) was a previous local contribution to C(m, n)
 * This function returns k' such that GEMM(m, n, k') is the next
 * local GEMM to execute
 * Returns -1 if there is no such k'
 */
int gemm_plan_next(gemm_plan_t *plan, int m, int n, int k);

END_C_DECLS

#endif /* _DPLASMA_BCAST_H_ */
