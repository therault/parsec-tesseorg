/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */
#include "dague_internal.h"
#include <cblas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

struct zlascal_args_s {
    dague_complex64_t    alpha;
    tiled_matrix_desc_t *descA;
};
typedef struct zlascal_args_s zlascal_args_t;

static int
dplasma_zlascal_operator( struct dague_execution_unit *eu,
                         void *_A,
                         void *op_data, ... )
{
    va_list ap;
    PLASMA_enum uplo;
    int i, m, n;
    int tempmm, tempnn, ldam;
    tiled_matrix_desc_t *descA;
    dague_complex64_t alpha;
    zlascal_args_t *args = (zlascal_args_t*)op_data;
    dague_complex64_t *A = (dague_complex64_t*)_A;
    (void)eu;

    va_start(ap, op_data);
    uplo = va_arg(ap, PLASMA_enum);
    m    = va_arg(ap, int);
    n    = va_arg(ap, int);
    va_end(ap);

    descA = args->descA;
    alpha = args->alpha;

    tempmm = ((m)==((descA->mt)-1)) ? ((descA->m)-(m*(descA->mb))) : (descA->mb);
    tempnn = ((n)==((descA->nt)-1)) ? ((descA->n)-(n*(descA->nb))) : (descA->nb);
    ldam = BLKLDD( *descA, m );

    /* Overwrite uplo when outside the diagonal */
    if (m != n) {
        uplo = PlasmaUpperLower;
    }

    switch ( uplo ) {
    case PlasmaUpper:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( dplasma_imin( i+1, tempmm ), CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;

    case PlasmaLower:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( dplasma_imax( tempmm, tempmm-i ), CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;
    default:
        for(i=0; i<tempnn; i++) {
            cblas_zscal( tempmm, CBLAS_SADDR(alpha), A+i*ldam, 1 );
        }
        break;
    }

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zlascal_New - Generates the object that scales a matrix by a given scalar.
 *
 * See dplasma_map_New() for further information.
 *
 *  WARNINGS:
 *      - The computations are not done by this call.
 *      - This routine is equivalent to the pzlascal routine from ScaLapack and
 *  not to the zlascl/pzlascl routines from Lapack/ScaLapack.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *          = PlasmaUpperHessenberg: A is an upper Hessenberg matrix (Not
 *          supported)
 *
 * @param[in] alpha
 *          The scalar to use to scale the matrix.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, the matrix A is scaled by alpha.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlascal_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal
 * @sa dplasma_zlascal_Destruct
 * @sa dplasma_clascal_New
 * @sa dplasma_dlascal_New
 * @sa dplasma_slascal_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zlascal_New( PLASMA_enum uplo,
                     dague_complex64_t alpha,
                     tiled_matrix_desc_t *A )
{
    zlascal_args_t *params = (zlascal_args_t*)malloc(sizeof(zlascal_args_t));

    params->alpha = alpha;
    params->descA = A;

    return dplasma_map_New( uplo, A, dplasma_zlascal_operator, params );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zlascal_Destruct - Free the data structure associated to an object
 *  created with dplasma_zlascal_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal_New
 * @sa dplasma_zlascal
 *
 ******************************************************************************/
void
dplasma_zlascal_Destruct( dague_object_t *o )
{
    dplasma_map_Destruct( o );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 * dplasma_zlascal - Scales a matrix by a given scalar.
 *
 * See dplasma_map() for further information.
 *
 *  WARNINGS:
 *      - This routine is equivalent to the pzlascal routine from ScaLapack and
 *  not to the zlascl/pzlascl routines from Lapack/ScaLapack.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies which part of matrix A is set:
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *          = PlasmaUpperHessenberg: A is an upper Hessenberg matrix (Not
 *          supported)
 *
 * @param[in] alpha
 *          The scalar to use to scale the matrix.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A. Any tiled matrix
 *          descriptor can be used.
 *          On exit, the matrix A is scaled by alpha.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlascal_New
 * @sa dplasma_zlascal_Destruct
 * @sa dplasma_clascal
 * @sa dplasma_dlascal
 * @sa dplasma_slascal
 *
 ******************************************************************************/
int
dplasma_zlascal( dague_context_t     *dague,
                 PLASMA_enum          uplo,
                 dague_complex64_t    alpha,
                 tiled_matrix_desc_t *A )
{
    dague_object_t *dague_zlascal = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) &&
        (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        dplasma_error("dplasma_zlascal", "illegal value of type");
        return -2;
    }

    dague_zlascal = dplasma_zlascal_New(uplo, alpha, A);

    if ( dague_zlascal != NULL ) {
        dague_enqueue(dague, (dague_object_t*)dague_zlascal);
        dplasma_progress(dague);
        dplasma_zlascal_Destruct( dague_zlascal );
    }
    return 0;
}
