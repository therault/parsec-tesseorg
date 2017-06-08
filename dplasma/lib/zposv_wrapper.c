/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zposv - Computes the solution to a system of linear equations A * X =
 * B, where A is an N-by-N symmetric positive definite (or Hermitian positive
 * definite in the complex case) matrix and X and B are N-by-NRHS matrices.  The
 * Cholesky decomposition is used to factor A as
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo = PlasmaUpper} \f]
 *
 * where U is an upper triangular matrix and  L is a lower triangular matrix.
 * The factored form of A is then used to solve the system of equations A * X = B.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 * @param[in,out] B
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if return value = 0, B is overwritten by the solution matrix X.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zposv_New
 * @sa dplasma_zposv_Destruct
 * @sa dplasma_cposv
 * @sa dplasma_dposv
 * @sa dplasma_sposv
 *
 ******************************************************************************/
int
dplasma_zposv( parsec_context_t *parsec,
               PLASMA_enum uplo,
               tiled_matrix_desc_t *A,
               tiled_matrix_desc_t *B )
{
    int info;

    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zposv", "illegal value of uplo");
        return -1;
    }

#ifdef PARSEC_COMPOSITION
    parsec_handle_t *parsec_ztrsm1 = NULL;
    parsec_handle_t *parsec_ztrsm2 = NULL;
    parsec_handle_t *parsec_zpotrf = NULL;

    parsec_zpotrf = dplasma_zpotrf_New(uplo, A, &info);
    if ( uplo == PlasmaUpper ) {
      parsec_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B);
      parsec_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B);
    } else {
      parsec_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B);
      parsec_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B);
    }

    parsec_enqueue( parsec, parsec_zpotrf );
    parsec_enqueue( parsec, parsec_ztrsm1 );
    parsec_enqueue( parsec, parsec_ztrsm2 );

    dplasma_progress( parsec );

    dplasma_zpotrf_Destruct( parsec_zpotrf );
    dplasma_ztrsm_Destruct( parsec_ztrsm1 );
    dplasma_ztrsm_Destruct( parsec_ztrsm2 );
#else
    info = dplasma_zpotrf( parsec, uplo, A);
    if ( info == 0 ) {
      if ( uplo == PlasmaUpper ) {
        dplasma_ztrsm( parsec, PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B );
        dplasma_ztrsm( parsec, PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B );
      } else {
        dplasma_ztrsm( parsec, PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B );
        dplasma_ztrsm( parsec, PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B );
      }
    }
#endif
    return info;
}
