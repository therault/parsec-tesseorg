/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _zsumma_h_has_been_included_
#define _zsumma_h_has_been_included_

#include "irregular_tiled_matrix.h"

#if defined(PARSEC_HAVE_RECURSIVE)
#include "parsec/recursive.h"
#endif

BEGIN_C_DECLS

/***********************************************************
 *               Blocking interface
 */
/* Level 3 Blas */
int summa_zsumma( parsec_context_t *parsec,
                  PLASMA_enum transA, PLASMA_enum transB,
                  parsec_complex64_t alpha,
				  const irregular_tiled_matrix_desc_t *A,
                  const irregular_tiled_matrix_desc_t *B,
                  irregular_tiled_matrix_desc_t *C);

/* Recursive kernel */
int
summa_zsumma_rec( parsec_context_t *parsec,
				  PLASMA_enum transA, PLASMA_enum transB,
				  parsec_complex64_t alpha,
				  const irregular_tiled_matrix_desc_t *A,
				  const irregular_tiled_matrix_desc_t *B,
				  irregular_tiled_matrix_desc_t *C, int bigtile, int opttile);

/***********************************************************
 *             Non-Blocking interface
 */
/* Level 3 Blas */
parsec_handle_t*
summa_zsumma_New( PLASMA_enum transA, PLASMA_enum transB,
                  parsec_complex64_t alpha, const irregular_tiled_matrix_desc_t* A,
                  const irregular_tiled_matrix_desc_t* B,
                  irregular_tiled_matrix_desc_t* C);

/***********************************************************
 *               Destruct functions
 */
/* Level 3 Blas */
void summa_zsumma_Destruct( parsec_handle_t *o );

void summa_zsumma_recursive_Destruct(parsec_handle_t *handle);

/**********************************************************
 * Check routines
 */
/* int check_zsumma(  parsec_context_t *parsec, int loud, PLASMA_enum uplo, irregular_tiled_matrix_desc_t *A, irregular_tiled_matrix_desc_t *b, irregular_tiled_matrix_desc_t *x ); */

#if defined(PARSEC_HAVE_RECURSIVE)
void summa_zsumma_setrecursive( parsec_handle_t *o, int bigtile, int opttile );

static inline int summa_recursivecall_callback(parsec_handle_t* parsec_handle, void* cb_data)
{
    int i, rc = 0;
    cb_data_t* data = (cb_data_t*)cb_data;

    rc = __parsec_complete_execution(data->eu, data->context);

    for(i=0; i<data->nbdesc; i++){
        irregular_tiled_matrix_desc_destroy( (irregular_tiled_matrix_desc_t*)(data->desc[i]) );
        free( data->desc[i] );
    }

    data->destruct( parsec_handle );
    free(data);

    return rc;
}
#endif

END_C_DECLS

#endif /* _zsumma_h_has_been_included_ */
