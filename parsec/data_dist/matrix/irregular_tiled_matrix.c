/*
 * Copyright (c) 2016-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <math.h>

#include "parsec/parsec_config.h"
#include "parsec/devices/device.h"
#include "parsec/parsec_internal.h"
//#include "parsec/debug.h"
#include "parsec/vpmap.h"
#include "parsec.h"
#include "parsec/data.h"
//include "dplasma.h"
#include "parsec/data_dist/matrix/irregular_tiled_matrix.h"
#include "parsec/data_dist/matrix/irregular_subtile.h"

#ifdef PARSEC_HAVE_MPI
#include <mpi.h>
#endif /* PARSEC_HAVE_MPI */

static uint32_t       irregular_tiled_matrix_rank_of(     parsec_data_collection_t* dc, ...);
static uint32_t       irregular_tiled_matrix_rank_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static int32_t        irregular_tiled_matrix_vpid_of(     parsec_data_collection_t* dc, ...);
static int32_t        irregular_tiled_matrix_vpid_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_t* irregular_tiled_matrix_data_of(     parsec_data_collection_t* dc, ...);
static parsec_data_t* irregular_tiled_matrix_data_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_key_t irregular_tiled_matrix_coord_to_key(parsec_data_collection_t* dc, ...);
static void           irregular_tiled_matrix_key_to_coord(parsec_data_collection_t* dc, parsec_data_key_t key, int *i, int *j);

static void irregular_tile_data_copy_construct(irregular_tile_data_copy_t* t)
{
    (void)t;
#if defined(PARSEC_DEBUG_PARANOID)
#    if defined(PARSEC_DEBUG)
    t->magic = LET_THE_MAGIC_HAPPENS;
#    endif
    t->mb = -1;
    t->nb = -1;
#endif
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Allocate irregular tile data copy %p", t);
}

static void irregular_tile_data_copy_destruct(irregular_tile_data_copy_t* tile)
{
    parsec_data_copy_t *obj = (parsec_data_copy_t*)tile;
    (void)obj;
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "Release irregular tile data copy %p", obj);
    /* nothing to erase in this type */
}

OBJ_CLASS_INSTANCE(irregular_tile_data_copy_t, parsec_data_copy_t,
                   irregular_tile_data_copy_construct,
                   irregular_tile_data_copy_destruct);


parsec_data_t*
irregular_tile_data_create( parsec_data_t **holder,
                            irregular_tiled_matrix_desc_t *desc,
                            parsec_data_key_t key, void *ptr,
                            int mb, int nb, size_t elem_size )
{
    parsec_data_t *data = *holder;

    if( NULL == data ) {
        data = OBJ_NEW(parsec_data_t);
        data->owner_device = 0;
        data->key = key;
        data->dc = (parsec_data_collection_t*)desc;
        data->nb_elts = mb*nb*elem_size;
        /* data->nb_elts = desc->max_tile; */

        parsec_data_copy_t* data_copy = (parsec_data_copy_t*)OBJ_NEW(irregular_tile_data_copy_t);
        parsec_data_copy_attach(data, data_copy, 0);
        data_copy->device_private = ptr;

        irregular_tile_data_copy_t *t = (irregular_tile_data_copy_t *)data_copy;
        t->mb = mb;
        t->nb = nb;
        /* This happens while inserting tiles in ddesc before the parallel execution */
        *holder = data;
    }
    else {
        /* Do we have a copy of this data */
        if( NULL == data->device_copies[0] ) {
            parsec_data_copy_t* data_copy = parsec_data_copy_new(data, 0);
            data_copy->device_private = ptr;
            irregular_tile_data_copy_t *t = (irregular_tile_data_copy_t *)data_copy;
            t->mb = mb;
            t->nb = nb;
        }
    }
    assert( data->key == key );
    return data;
}

static irregular_tile_t* get_tile(irregular_tiled_matrix_desc_t* desc, int i, int j)
{
    int pos;
    i += desc->i;
    j += desc->j;
    /* Row major column storage */
    pos = (desc->lnt * i) + j;
    assert(0 <= pos && pos < desc->lmt * desc->lnt);
    return desc->data_map+pos;
}

static parsec_data_t* get_data(irregular_tiled_matrix_desc_t* desc, int i, int j)
{
    int pos;
    assert(0 == desc->i);
    assert(0 == desc->j);
    /* Row major column storage */
    pos = (desc->lnt * i) + j;
    assert(0 <= pos && pos < desc->lmt * desc->lnt);
    return desc->local_data_map[pos];
}

static uint32_t irregular_tiled_matrix_rank_of(parsec_data_collection_t* dc, ...)
{
    int i, j;
    va_list ap;

    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)dc;

    va_start(ap, dc);
    i = va_arg(ap, int);
    j = va_arg(ap, int);
    va_end(ap);

    irregular_tile_t* t = get_tile(desc, i, j);

    assert(NULL != t);
    return t->rank;
}

static int32_t irregular_tiled_matrix_vpid_of(parsec_data_collection_t* dc, ...)
{
    int i, j;
    va_list ap;

    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)dc;

    va_start(ap, dc);
    i = va_arg(ap, int);
    j = va_arg(ap, int);
    va_end(ap);

    irregular_tile_t* t = get_tile(desc, i, j);

    assert(NULL != t);
    return t->vpid;
}

static parsec_data_t* irregular_tiled_matrix_data_of(parsec_data_collection_t* dc, ...)
{
    int i, j;
    va_list ap;

    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)dc;

    va_start(ap, dc);
    i = va_arg(ap, int);
    j = va_arg(ap, int);
    va_end(ap);

#if defined(DISTRIBUTED)
    assert(dc->myrank == irregular_tiled_matrix_rank_of(dc, i, j));
#endif

    parsec_data_t* t = get_data(desc, i, j);
    assert(NULL != t);
    return t;
}

static void irregular_tiled_matrix_key_to_coord(parsec_data_collection_t* dc, parsec_data_key_t key, int *i, int *j)
{
    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)dc;
    *i = key / desc->lnt;
    *j = key % desc->lnt;
}

static uint32_t irregular_tiled_matrix_rank_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(dc, key, &i, &j);
    return irregular_tiled_matrix_rank_of(dc, i, j);
}

static int32_t irregular_tiled_matrix_vpid_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(dc, key, &i, &j);
    return irregular_tiled_matrix_vpid_of(dc, i, j);
}

static parsec_data_t* irregular_tiled_matrix_data_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(dc, key, &i, &j);
    return irregular_tiled_matrix_data_of(dc, i, j);
}

static parsec_data_key_t irregular_tiled_matrix_coord_to_key(struct parsec_data_collection_s *dc, ...)
{
    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)dc;
    int i, j;
    va_list ap;

    va_start(ap, dc);
    i = va_arg(ap, unsigned int);
    j = va_arg(ap, unsigned int);
    va_end(ap);

    i += desc->i;
    j += desc->j;

    parsec_data_key_t k = (i * desc->lnt) + j;

    return k;
}

static char *irregular_tiled_matrix_key_to_string(parsec_data_collection_t *dc, parsec_data_key_t key, char * buffer, uint32_t buffer_size)
{
    unsigned int m, n;
    int res;
    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)dc;

    m = key % desc->lnt;
    n = key / desc->lnt;
    res = snprintf(buffer, buffer_size, "%s(%u, %u)", dc->dc_name, m, n);
    if (res < 0)
        parsec_warning("Wrong key_to_string for tile (%u, %u) key: %u", m, n, key);
    return buffer;
}


int tile_is_local(int i, int j, grid_2Dcyclic_t* g)
{
    return (((i/g->strows)%g->rows == g->rrank)&&((j/g->strows)%g->cols == g->crank));
}

unsigned int tile_owner(int i, int j, grid_2Dcyclic_t* g)
{
    int rows_tile = g->strows*g->rows;
    int cols_tile = g->stcols*g->cols;

    int iP = i%rows_tile;
    int iQ = j%cols_tile;

    iP /= g->strows;
    iQ /= g->stcols;

    return iP*g->cols+iQ;
}

int get_tile_count(const irregular_tiled_matrix_desc_t *desc, int m, int n)
{
    assert(0 <= m && m < desc->mt && 0 <= n && n < desc->nt);
    uint64_t res = desc->Mtiling[m] * desc->Ntiling[n];
    uint64_t max = (uint64_t)desc->m * (uint64_t)desc->n;
    (void)res;
    (void)max;
    assert(0 < res && res <= max);
    return res;
}

void irregular_tiled_matrix_desc_init(irregular_tiled_matrix_desc_t* ddesc,
                                      enum matrix_type mtype,
                                      unsigned int nodes, unsigned int myrank,
                                      unsigned int lm, unsigned int ln,
                                      unsigned int lmt, unsigned int lnt,
                                      unsigned int* Mtiling, unsigned int* Ntiling,
                                      unsigned int i, unsigned int j,
                                      unsigned int mt, unsigned int nt,
                                      unsigned int P,
                                      void *(*future_resolve_fct)(void*, void *, void *))
{
    unsigned int k;
    parsec_data_collection_t *d = (parsec_data_collection_t*)ddesc;

    if(nodes < P)
        parsec_fatal("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d", nodes, P);

    int Q = nodes / P;
    if(nodes != P*Q)
        parsec_warning("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d", nodes, P, Q);

    parsec_data_collection_init(d, nodes, myrank);

    d->rank_of     = irregular_tiled_matrix_rank_of;
    d->rank_of_key = irregular_tiled_matrix_rank_of_key;
    d->vpid_of     = irregular_tiled_matrix_vpid_of;
    d->vpid_of_key = irregular_tiled_matrix_vpid_of_key;
    d->data_of     = irregular_tiled_matrix_data_of;
    d->data_of_key = irregular_tiled_matrix_data_of_key;
    d->data_key    = irregular_tiled_matrix_coord_to_key;

    d->key_to_string = irregular_tiled_matrix_key_to_string;

    grid_2Dcyclic_init(&ddesc->grid, myrank, P, Q, 1, 1);

    ddesc->data_map = (irregular_tile_t*)calloc(lmt*lnt, sizeof(irregular_tile_t));
    ddesc->local_data_map = (parsec_data_t**)calloc(lmt*lnt, sizeof(parsec_data_t*));

    ddesc->Mtiling = (unsigned int*)malloc(lmt*sizeof(unsigned int));
    ddesc->Ntiling = (unsigned int*)malloc(lnt*sizeof(unsigned int));

    for (k = 0; k < lmt; ++k) ddesc->Mtiling[k] = Mtiling[k];
    for (k = 0; k < lnt; ++k) ddesc->Ntiling[k] = Ntiling[k];

    ddesc->mtype = mtype;
    ddesc->storage = matrix_Tile;
    ddesc->dtype = irregular_tiled_matrix_desc_type;
    ddesc->bsiz = lm*ln;
    ddesc->lm = lm;
    ddesc->ln = ln;
    ddesc->lmt = lmt;
    ddesc->lnt = lnt;
    ddesc->mt = mt;
    ddesc->nt = nt;
    ddesc->i = i;
    ddesc->j = j;

    ddesc->m = 0;
    for (k = 0; k < lmt; ++k) ddesc->m += Mtiling[k];
    ddesc->n = 0;
    for (k = 0; k < lnt; ++k) ddesc->n += Ntiling[k];
    ddesc->lm = 0;
    for (k = 0; k < mt; ++k) ddesc->lm += Mtiling[ddesc->i+k];
    ddesc->ln = 0;
    for (k = 0; k < nt; ++k) ddesc->ln += Ntiling[ddesc->j+k];

    ddesc->nb_local_tiles = 0;
    ddesc->max_mb = 0;
    ddesc->max_tile = 0;

    for (i = 0; i < lmt; ++i) {
        if (Mtiling[i] > ddesc->max_mb)
            ddesc->max_mb = Mtiling[i];
        for (j = 0; j < lnt; ++j)
            if (Mtiling[i]*Ntiling[j] > ddesc->max_tile)
                /* Worst case scenario */
                ddesc->max_tile = Mtiling[i]*Ntiling[j];
    }
    ddesc->future_resolve_fct = future_resolve_fct;
    asprintf(&ddesc->super.dc_dim, "(%d, %d)", mt, nt);
}

void irregular_tiled_matrix_destroy_data(irregular_tiled_matrix_desc_t* ddesc)
{
    int i, j;
    for (i = 0; i < ddesc->lmt; ++i)
        for (j = 0; j < ddesc->lnt; ++j)
            if (ddesc->super.myrank == irregular_tiled_matrix_rank_of(&ddesc->super, i, j)) {
                uint32_t idx = ((parsec_data_collection_t*)ddesc)->data_key((parsec_data_collection_t*)ddesc, i, j);
                parsec_data_destroy(ddesc->local_data_map[idx]);
            }
}

void irregular_tiled_matrix_desc_destroy(irregular_tiled_matrix_desc_t* ddesc)
{
    irregular_tiled_matrix_destroy_data(ddesc);

    if (ddesc->data_map)       free(ddesc->data_map);
    if (ddesc->local_data_map) free(ddesc->local_data_map);
    if (ddesc->Mtiling)        free(ddesc->Mtiling);
    if (ddesc->Ntiling)        free(ddesc->Ntiling);

    parsec_data_collection_destroy((parsec_data_collection_t*)ddesc);
}

/* sets up the tile by constructing a new object, then filling specific fields with input parameter */
 void irregular_tiled_matrix_desc_set_data(irregular_tiled_matrix_desc_t *ddesc, void *actual_data, uint32_t idx, int mb, int nb, int vpid, int rank)
{
    (void)actual_data;
    (void)idx;
    (void)vpid;
    (void)rank;

    if (NULL != actual_data) {
        parsec_data_create(ddesc->local_data_map+idx,(parsec_data_collection_t*)ddesc, idx, actual_data,
                           mb*nb*parsec_datadist_getsizeoftype(ddesc->mtype));
        ddesc->nb_local_tiles++;
    }
    ddesc->data_map[idx].vpid = vpid;
    ddesc->data_map[idx].rank = rank;

    assert ((uint32_t)rank == ((parsec_data_collection_t*)ddesc)->myrank || actual_data == NULL);
}

#if 0
int
summa_aux_getSUMMALookahead(irregular_tiled_matrix_desc_t *ddesc)
{
    /**
     * Assume that the number of threads per node is constant, and compute the
     * look ahead based on the global information to get the same one on all
     * nodes.
     */
    int nbunits = vpmap_get_nb_total_threads() * ddesc->super.nodes;
    double alpha =  9. * (double)nbunits / ( ddesc->mt * ddesc->nt );
/*
    printf("Look ahead of (%dx%d) with %d units is max of %g, %d: %d\n", 
           ddesc->mt, ddesc->nt, nbunits,
           alpha, 2, summa_imax( ceil( alpha ), 2 ));
*/
    if ( ddesc->super.nodes == 1 ) {
        /* No look ahaead */
        return summa_imax( ddesc->mt, ddesc->nt );
    }
    else {
        /* Look ahead of at least 2, and that provides 3 tiles per computational units */
        return summa_imax( ceil( alpha ), 2 );
    }
}
#endif
