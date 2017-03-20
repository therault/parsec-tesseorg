/*
 * Copyright (c) 2017      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __IRREGULAR_SUBTILE_H__
#define __IRREGULAR_SUBTILE_H__

<<<<<<< HEAD
<<<<<<< HEAD
#include "irregular_tiled_matrix.h"
=======
#include "summa/irregular_tiled_matrix.h"
>>>>>>> Adds: recursive tiling for irregular tiling
=======
#include "irregular_tiled_matrix.h"
>>>>>>> Temp commit

BEGIN_C_DECLS

/**
 * Exploit the irregular_tiled_matrix_desc to recursively split a single tile part of a
 * irregular_tiled_matrix_desc_t
 */
typedef struct irregular_subtile_desc_s {
    irregular_tiled_matrix_desc_t super;
    void *mat;      /**< pointer to the beginning of the matrix */
    int vpid;
    int i, j;
} irregular_subtile_desc_t;

/**
 * Initialize a descriptor to apply a recursive call on a single tile of a more
 * general tile descriptor.
 *
 * @param[in] tdesc
 *        irregular_tiled_matrix_descriptor which owns the tile that will be split into
 *        smaller tiles.
 *
 * @param[in] mt
 *        Row coordinate of the tile to split into the larger matrix.
 *
 * @param[in] nt
 *        Column coordinate of the tile to split into the larger matrix.
 *
 * @param[in] mb
 *        Number of rows in each subtiles
 *
 * @param[in] nb
 *        Number of columns in each subtiles
 *
 * @param[in] i
 *        Row index of the first element of the submatrix. 0 being the first
 *        row of the original tile.
 *
 * @param[in] j
 *        Column index of the first element of the submatrix. 0 being the first
 *        row of the original tile.
 *
 * @param[in] m
 *        Number of rows in the submatrix.
 *
 * @param[in] n
 *        Number of columns in the submatrix.
 *
 * @return
 *       Descriptor of the tile (mt, nt) of tdesc split in tiles of size mb by
 *       nb.
 *
 */
irregular_subtile_desc_t *irregular_subtile_desc_create( const irregular_tiled_matrix_desc_t *tdesc,
                                                         int mm, int nn, int mt, int nt);   /* Tile in tdesc */

two_dim_block_cyclic_t* recursive_fake_Cdist(const two_dim_block_cyclic_t* original);

END_C_DECLS

#endif /* __IRREGULAR_SUBTILE_H__*/
