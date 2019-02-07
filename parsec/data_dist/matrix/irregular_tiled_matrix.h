#ifndef _TILE_COLLECTION_H_
#define _TILE_COLLECTION_H_

#include <assert.h>
#include "parsec/parsec_config.h"
#include "parsec/data.h"
#include "parsec/data_internal.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/grid_2Dcyclic.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
//#include "dplasma/include/dplasma_complex.h"

BEGIN_C_DECLS

/* ColMajor, RowMajor, be careful */
#define SUMMA_BLKLDD( _desc_, _i_ ) ( (_desc_)->storage == matrix_Tile ? (_desc_)->Mtiling[_i_] : (_desc_)->llm )
/* Could be the oposite... */
#define SUMMA_N_BLKLDD( _desc_, _i_ ) ( (_desc_)->storage == matrix_Tile ? (_desc_)->Mtiling[_i_] : (_desc_)->llm )
#define SUMMA_T_BLKLDD( _desc_, _i_ ) ( (_desc_)->storage == matrix_Tile ? (_desc_)->Ntiling[_i_] : (_desc_)->lln )

#define SUMMA_N_ROWS( _desc_, _i_ ) ( (_desc_)->storage == matrix_Tile ? (_desc_)->Mtiling[_i_] : (_desc_)->llm )
#define SUMMA_N_COLS( _desc_, _i_ ) ( (_desc_)->storage == matrix_Tile ? (_desc_)->Ntiling[_i_] : (_desc_)->lln )
#define SUMMA_T_ROWS( _desc_, _i_ ) ( (_desc_)->storage == matrix_Tile ? (_desc_)->Ntiling[_i_] : (_desc_)->lln )
#define SUMMA_T_COLS( _desc_, _i_ ) ( (_desc_)->storage == matrix_Tile ? (_desc_)->Mtiling[_i_] : (_desc_)->llm )

#define irregular_tiled_matrix_desc_type 0x10
#define LET_THE_MAGIC_HAPPENS 1234567890


typedef struct irregular_tile_s {
    int                rank;
    int                vpid;
} irregular_tile_t;

typedef struct irregular_tile_data_copy_s {
    parsec_data_copy_t  super;
#if defined(PARSEC_DEBUG)
    uint64_t           magic;          /**< constant to assert cast went as planned */
#endif
    int                mb;             /**< number of rows in a tile */
    int                nb;             /**< number of columns in a tile */
} irregular_tile_data_copy_t;

/* Declare this struct as a class */
/* In .c inheritance will be declared */
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(irregular_tile_data_copy_t);


typedef struct irregular_tiled_matrix_desc_s {
    parsec_data_collection_t     super;           /**< inherited class */
    grid_2Dcyclic_t              grid;            /**< processes grid */
    irregular_tile_t            *data_map;        /**< map of the meta data of tiles */
    parsec_data_t              **local_data_map;  /**< map of the local data */
    unsigned int                *Mtiling;         /**< array of size lmt giving the tile size */
    unsigned int                *Ntiling;         /**< array of size lnt giving the tile size */
    enum matrix_type             mtype;           /**< precision of the matrix */
    enum matrix_storage          storage;    /**< storage of the matrix   */
    int                          dtype;           /**< Distribution type of descriptor */
    int                          bsiz;            /**< size in elements incl padding of a tile - derived parameter */
    unsigned int                 lm;              /**< number of rows of the entire matrix */
    unsigned int                 ln;              /**< number of columns of the entire matrix */
    unsigned int                 llm;
    unsigned int                 lln;
    int                          lmt;             /**< number of tile rows of the entire matrix */
    int                          lnt;             /**< number of tile columns of the entire matrix */
    int                          i;               /**< row tile index to the beginning of the submatrix */
    int                          j;               /**< column tile index to the beginning of the submatrix */
    int                          m;               /**< number of rows of the submatrix - derived parameter */
    int                          n;               /**< number of columns of the submatrix - derived parameter */
    int                          mt;              /**< number of tile rows of the submatrix */
    int                          nt;              /**< number of tile columns of the submatrix */
    int                          nb_local_tiles;  /**< number of tile handled locally */
    unsigned int                 max_mb;          /**< maximum value of mb */
    unsigned int                 max_tile;        /**< size of the biggest tile */
    void *(*future_resolve_fct)(void*,void*,void*); /**< Function to use to resolve future if this is needed */
} irregular_tiled_matrix_desc_t;

/* Workaround : irregular_tile_data_copy_t* irregular_tile_data_copy_create( irregular_tile_data_copy_t **holder, */
/*                                                              parsec_data_key_t key, int owner, int mb, int nb); */
parsec_data_copy_t* irregular_tile_data_copy_create( parsec_data_copy_t **holder,
                                                     parsec_data_key_t key, int owner, int mb, int nb);

parsec_data_t* irregular_tile_data_create(parsec_data_t **holder,
                                          irregular_tiled_matrix_desc_t *desc,
                                          parsec_data_key_t key, void *ptr, int mb, int nb, size_t elem_size);

int get_tile_count(const irregular_tiled_matrix_desc_t *desc, int m, int n);

int tile_is_local(int i, int j, grid_2Dcyclic_t* g);

unsigned int tile_owner(int i, int j, grid_2Dcyclic_t* g);

void export_pythons(irregular_tiled_matrix_desc_t *A, irregular_tiled_matrix_desc_t *B, irregular_tiled_matrix_desc_t *C, int P, int Q, int N, int NB, int MB, int psize);

void irregular_tiled_matrix_desc_init(
    irregular_tiled_matrix_desc_t* dc,
    enum matrix_type mtype,
    unsigned int nodes, unsigned int myrank,
    /* global number of rows/cols */
    unsigned int lm, unsigned int ln,
    /* global number of tiles */
    unsigned int lmt, unsigned int lnt,
    /* tiling of the submatrix */
    unsigned int*Mtiling, unsigned int* Ntiling,
    /* first tile of the submatrix */
    unsigned int i, unsigned int j,
    /* number of tiles of the submatrix */
    unsigned int mt, unsigned int nt,
    unsigned int P,
    void *(*future_resolve_fct)(void*,void*,void*));

typedef int (*irregular_tiled_matrix_init_op_t)( struct parsec_execution_stream_s *es,
                                                 irregular_tiled_matrix_desc_t *desc1,
                                                 int m, int n,
                                                 void *args );


/* add a parameter for number of expected tiles to register?*/
/* I could then do a collective operation to to build the structure of the matrix */
/* and the tiling vectors can be infered by sharing information */

void irregular_tiled_matrix_desc_destroy(irregular_tiled_matrix_desc_t* dc);

/* i, j are tile coordinates, nb, mb are tile sizes */
void irregular_tiled_matrix_desc_set_data(irregular_tiled_matrix_desc_t *dc, void *actual_data, uint32_t idx, int mb, int nb, int vpid, int rank);

void irregular_tiled_matrix_desc_build(irregular_tiled_matrix_desc_t *dc);

//int summa_aux_getSUMMALookahead( irregular_tiled_matrix_desc_t *dc );

END_C_DECLS

#endif /* _TILE_COLLECTION_H_ */
