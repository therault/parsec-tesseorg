


#ifndef _TILE_COLLECTION_H_
#define _TILE_COLLECTION_H_

#include "dague_config.h"
#include <assert.h>
#include "dague/data_distribution.h"
#include "dague/data.h"
#include "dague/data_internal.h"
#include "data_dist/matrix/grid_2Dcyclic.h"
#include "precision.h"
#include "dague/data.h"

BEGIN_C_DECLS

enum tile_coll_type {
    tile_coll_Byte          = 0, /**< unsigned char  */
    tile_coll_Integer       = 1, /**< signed int     */
    tile_coll_RealFloat     = 2, /**< float          */
    tile_coll_RealDouble    = 3, /**< double         */
    tile_coll_ComplexFloat  = 4, /**< complex float  */
    tile_coll_ComplexDouble = 5  /**< complex double */
};

enum tile_coll_storage {
    tile_coll_Lapack        = 0, /**< LAPACK Layout or Column Major  */
    tile_coll_Tile          = 1, /**< Tile Layout or Column-Column Rectangular Block (CCRB) */
};

/**
 * Put our own definition of Upper/Lower/General values mathing the
 * Cblas/Plasma/... ones to avoid teh dependency
 */
enum tile_coll_uplo {
    tile_coll_Upper      = 121,
    tile_coll_Lower      = 122,
    tile_coll_UpperLower = 123
};

static inline int dague_irreguler_tiled_matrix_getsizeoftype(enum tile_coll_type type)
{
    switch( type ) {
    case tile_coll_Byte          : return sizeof(char);
    case tile_coll_Integer       : return sizeof(int);
    case tile_coll_RealFloat     : return sizeof(float);
    case tile_coll_RealDouble    : return sizeof(double);
    case tile_coll_ComplexFloat  : return sizeof(dague_complex32_t);
    case tile_coll_ComplexDouble : return sizeof(dague_complex64_t);
    default:
        return -1;
    }
}

/**
 * Convert from a matrix type to a more traditional PaRSEC type usable for
 * creating arenas.
 */
static inline int dague_translate_irregular_tiled_matrix_type( enum tile_coll_type mt, dague_datatype_t* dt )
{
    switch(mt) {
    case tile_coll_Byte:          *dt = dague_datatype_int8_t; break;
    case tile_coll_Integer:       *dt = dague_datatype_int32_t; break;
    case tile_coll_RealFloat:     *dt = dague_datatype_float_t; break;
    case tile_coll_RealDouble:    *dt = dague_datatype_double_t; break;
    case tile_coll_ComplexFloat:  *dt = dague_datatype_complex_t; break;
    case tile_coll_ComplexDouble: *dt = dague_datatype_double_complex_t; break;
    default:
        fprintf(stderr, "%s:%d Unknown tile_coll_type (%d)\n", __func__, __LINE__, mt);
        return -1;
    }
    return 0;
}

#define SUMMABLKLDD( _desc_, _m_ ) ( (_desc_).mb )

#define irregular_tiled_matrix_desc_type 0x10
#define LET_THE_MAGIC_HAPPENS 0xCAFECAFE


typedef struct irregular_tile_data_s {
	dague_data_t super;
#if defined(DAGUE_DEBUG)
	uint64_t magic;     /**< constant to assert cast went as planned */
#endif
    int           rank;
    int           vpid;

    int tileld;         /**< leading dimension of each tile (Should be a function depending on the row) */
    int mb;             /**< number of rows in a tile */
    int nb;             /**< number of columns in a tile */
	void *data;         /**< pointer to the tile data */
} irregular_tile_data_t;


typedef struct irregular_tiled_matrix_desc_s {
    dague_ddesc_t super;
	grid_2Dcyclic_t grid;
    irregular_tile_data_t**       data_map;   /**< map of the data */
	unsigned int * itiling; /**< array of size lmt+1 giving the range of indices of tile t in [itiling(t);itiling(t+1)]*/
	unsigned int * jtiling; /**< array of size lnt+1 giving the range of indices of tile t in [jtiling(t);jtiling(t+1)]*/
    enum tile_coll_type     mtype;      /**< precision of the matrix */
    enum tile_coll_storage  storage;    /**< storage of the matrix */
    int dtype;          /**< Distribution type of descriptor */
    int bsiz;           /**< size in elements including padding of a tile - derived parameter */
    int lm;             /**< number of rows of the entire matrix */
    int ln;             /**< number of columns of the entire matrix */
    int lmt;            /**< number of tile rows of the entire matrix */
    int lnt;            /**< number of tile columns of the entire matrix */
    int llm;            /**< number of rows of the matrix stored localy - derived parameter */
    int lln;            /**< number of columns of the matrix stored localy - derived parameter */
    int i;              /**< row tile index to the beginning of the submatrix */
    int j;              /**< column tile index to the beginning of the submatrix */
    int m;              /**< number of rows of the submatrix - derived parameter */
    int n;              /**< number of columns of the submatrix - derived parameter */
    int mt;             /**< number of tile rows of the submatrix */
    int nt;             /**< number of tile columns of the submatrix */
    int nb_local_tiles; /**< number of tile handled locally */
} irregular_tiled_matrix_desc_t;

DAGUE_DECLSPEC dague_data_t* irregular_tile_data_new(void);

int tile_is_local(int i, int j, grid_2Dcyclic_t* g);

void irregular_tiled_matrix_desc_init(
	irregular_tiled_matrix_desc_t* ddesc,
	enum tile_coll_type mtype,
	unsigned int nodes, unsigned int myrank,
	/* global number of rows/cols */
	unsigned int lm, unsigned int ln,
	/* global number of tiles */
	unsigned int lmt, unsigned int lnt,
	/* tiling of the submatrix */
	unsigned int*itiling, unsigned int* jtiling,
	/* first tile of the submatrix */
	unsigned int i, unsigned int j,
	/* number of tiles of the submatrix */
	unsigned int mt, unsigned int nt,
	unsigned int P);
/* add a parameter for number of expected tiles to register?*/
/* I could then do a collective operation to to build the structure of the matrix */
/* and the tiling vectors can be infered by sharing information */

void irregular_tiled_matrix_desc_destroy(irregular_tiled_matrix_desc_t* ddesc);

/* i, j are tile coordinates, nb, mb are tile sizes */
void irregular_tiled_matrix_desc_set_data(irregular_tiled_matrix_desc_t *ddesc, void *actual_data, int i, int j, int nb, int mb, int vpid, int rank);

void irregular_tiled_matrix_desc_build(irregular_tiled_matrix_desc_t *ddesc);

int summa_aux_getSUMMALookahead( irregular_tiled_matrix_desc_t *ddesc );

#  if 0
/* Alternative */
void irregular_tiled_matrix_desc_init(irregular_tiled_matrix_desc_t* ddesc,
		       enum tile_coll_type mtype,
		       unsigned int nodes, unsigned int myrank,
		       unsigned int lm, unsigned int ln,
		       unsigned int lmt, unsigned int lnt,
		       unsigned int i, unsigned int j,
		       int local_tiles);

/* when inserting the local_tiles-th element of the matrix, this call becomes a collective operation with information sharing to discover the tiling after insertion of the whole matrix */
void irregular_tiled_matrix_desc_set_data(irregular_tiled_matrix_desc_t *ddesc, void *actual_data, int i, int j, unsigned int mb, unsigned int nb, int vpid, int rank);

#  endif

#endif /* _TILE_COLLECTION_H_ */
