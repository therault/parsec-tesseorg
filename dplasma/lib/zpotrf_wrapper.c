/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <plasma.h>
#include <dague.h>
#include <scheduling.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"

#include "generated/zpotrf_rl.h"
#include "generated/zpotrf_ll.h"

dague_object_t* 
dplasma_zpotrf_New(const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, int* INFO)
{
    dague_zpotrf_rl_object_t* object;
    dague_remote_dep_datatype_t default_ddt;
    int pri_change = dplasma_aux_get_priority( "POTRF", ddescA );

    object = dague_zpotrf_rl_new( (dague_ddesc_t*)ddescA, 
                                  ddescA->nb, ddescA->nt, pri_change, uplo, INFO );
    
    dplasma_datatype_define_tile(MPI_DOUBLE_COMPLEX, ddescA->nb, &default_ddt);
    dague_arena_construct(object->arenas[DAGUE_zpotrf_rl_DEFAULT_ARENA], 
                          ddescA->nb*ddescA->nb*sizeof(Dague_Complex64_t), 
                          DAGUE_ARENA_ALIGNMENT_SSE, default_ddt);
    return (dague_object_t*)object;
}

int
dplasma_zpotrf( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA) 
{
    dague_object_t *dague_zpotrf = NULL;

    int info;
    dague_zpotrf = dplasma_zpotrf_New(uplo, ddescA, &info);

    dague_enqueue( dague, (dague_object_t*)dague_zpotrf);
    dague_progress(dague);

    return info;
}

dague_object_t* 
dplasma_zpotrf_rl_New(const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, int* INFO)
{
    dague_zpotrf_rl_object_t* object;
    dague_remote_dep_datatype_t default_ddt;
    int pri_change = dplasma_aux_get_priority( "POTRF", ddescA );

    object = dague_zpotrf_rl_new( (dague_ddesc_t*)ddescA, 
                                  ddescA->nb, ddescA->nt, pri_change, uplo, INFO );
    
    dplasma_datatype_define_tile(MPI_DOUBLE_COMPLEX, ddescA->nb, &default_ddt);
    dague_arena_construct(object->arenas[DAGUE_zpotrf_rl_DEFAULT_ARENA], 
                          ddescA->nb*ddescA->nb*sizeof(Dague_Complex64_t), 
                          DAGUE_ARENA_ALIGNMENT_SSE, default_ddt);
    return (dague_object_t*)object;
}

dague_object_t* 
dplasma_zpotrf_ll_New(const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, int* INFO)
{
    dague_zpotrf_ll_object_t* object;
    dague_remote_dep_datatype_t default_ddt;
    int pri_change = dplasma_aux_get_priority( "POTRF", ddescA );

    object = dague_zpotrf_ll_new( (dague_ddesc_t*)ddescA, 
                                  ddescA->nb, ddescA->nt, pri_change, uplo, INFO );
    
    dplasma_datatype_define_tile(MPI_DOUBLE_COMPLEX, ddescA->nb, &default_ddt);
    dague_arena_construct(object->arenas[DAGUE_zpotrf_ll_DEFAULT_ARENA], 
                          ddescA->nb*ddescA->nb*sizeof(Dague_Complex64_t), 
                          DAGUE_ARENA_ALIGNMENT_SSE, default_ddt);
    return (dague_object_t*)object;
}

