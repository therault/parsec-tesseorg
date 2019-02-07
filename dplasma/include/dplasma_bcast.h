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
#include "parsec/class/parsec_hash_table.h"
#include "parsec/data_dist/matrix/irregular_tiled_matrix.h"

BEGIN_C_DECLS

/* Helper structures and functions */

typedef struct gemm_plan_update_list_s {
    parsec_hash_table_item_t ht_item;
    int              nb;
    int              updates_order[1];
} gemm_plan_update_list_t;

struct gemm_plan_s {
    int myrank;
    int worldsize;
    int mt;
    int nt;
    int kt;
    irregular_tiled_matrix_desc_t *descC;
    parsec_hash_table_t local_k; /* local_k(m, n) is a gemm_plan_update_list_t* */
};

/* This will define a GEMM execution plan for the bcast_gemm interface */
typedef struct gemm_plan_s gemm_plan_t;

/*
 * Returns a key from the coordinate m, n in C
 */
parsec_key_t gemm_plan_make_key(gemm_plan_t *plan, int m, int n);

/*
 * Returns the highest rank that holds a B(k, n) that contributes to C(m, n)
 */
int gemm_plan_highest_rank(gemm_plan_t *plan, int m, int n);

/*
 * Let k_i 0 <= i <= K_{m,n} be the set such that GEMM(m, n, k_i) runs on
 * rank r; This function returns k_0 
 */
int gemm_plan_kfirst(gemm_plan_t *plan, int m, int n, int r);

/*
 * Let k_i 0 <= i <= K_{m,n} be the set such that GEMM(m, n, k_i) runs on
 * rank r; This function returns k_{K_{m,n}}
 */
int gemm_plan_klast(gemm_plan_t *plan, int m, int n, int r);

/*
 * Let k_i 0 <= i <= K_{m,n} be the set such that GEMM(m, n, k_i) runs on
 * rank r; Assuming k = k_i, this function returns returns k_{i+1}
 */
int gemm_plan_knext(gemm_plan_t *plan, int m, int n, int r, int k);

/*
 * Let k_i 0 <= i <= K_{m,n} be the set such that GEMM(m, n, k_i) runs on
 * rank r; Assuming k = k_i, this function returns returns k_{i-1}
 */
int gemm_plan_kprev(gemm_plan_t *plan, int m, int n, int r, int k);

/*
 * This function returns the first rank that has local contributions
 * to C(m, n)
 */
int gemm_plan_rank_first(gemm_plan_t *plan, int m, int n);
    
/*
 * This function returns the last rank that has local contributions
 * to C(m, n). The target should hold C(m, n).
 */
int gemm_plan_rank_last(gemm_plan_t *plan, int m, int n);

/*
 * This function returns the next rank that has local contributions
 * to C(m, n), knowing that the current contribution happened on r.
 */
int gemm_plan_rank_next(gemm_plan_t *plan, int m, int n, int r);

/*
 * This function returns the previous rank that had local contributions
 * to C(m, n), knowing that the current contribution happened on r.
 */
int gemm_plan_rank_prev(gemm_plan_t *plan, int m, int n, int r);

END_C_DECLS

#endif /* _DPLASMA_BCAST_H_ */
