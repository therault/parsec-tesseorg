
/*
 * Copyright (c) 2016      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/devices/device.h"
#include "dague/dague_internal.h"
#include "dague/debug.h"
#include "irregular_tiled_matrix.h"
#include "dague/vpmap.h"
#include "dague.h"
#include "dague/data.h"
#include "dplasma.h"

#include <math.h>

#ifdef DAGUE_HAVE_MPI
#include <mpi.h>
#endif /* DAGUE_HAVE_MPI */


static uint32_t      irregular_tiled_matrix_rank_of(      dague_ddesc_t* ddesc, ...);
static uint32_t      irregular_tiled_matrix_rank_of_key(  dague_ddesc_t* ddesc, dague_data_key_t key);

static int32_t       irregular_tiled_matrix_vpid_of(      dague_ddesc_t* ddesc, ...);
static int32_t       irregular_tiled_matrix_vpid_of_key(  dague_ddesc_t* ddesc, dague_data_key_t key);

static dague_data_t* irregular_tiled_matrix_data_of(      dague_ddesc_t* ddesc, ...);
static dague_data_t* irregular_tiled_matrix_data_of_key(  dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_t* irregular_tiled_matrix_C_data_of(    dague_ddesc_t* ddesc, ...);
static dague_data_t* irregular_tiled_matrix_C_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);

static uint32_t      irregular_tiled_matrix_data_key(     dague_ddesc_t *ddesc, ...);


static void irregular_tile_data_copy_construct(irregular_tile_data_copy_t* t)
{
#if defined(DAGUE_DEBUG_PARANOID)
#    if defined(DAGUE_DEBUG)
	t->magic = LET_THE_MAGIC_HAPPENS;
#    endif
	t->mb = -1;
	t->nb = -1;
#endif
	DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Allocate irregular tile data copy %p", t);
}

static void irregular_tile_data_copy_destruct(irregular_tile_data_copy_t* tile)
{
	dague_data_copy_t *obj = (dague_data_copy_t*)tile;
    DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Release irregular tile data copy %p", obj);
    /* nothing to erase in this type */
}

OBJ_CLASS_INSTANCE(irregular_tile_data_copy_t, dague_data_copy_t,
                   irregular_tile_data_copy_construct,
                   irregular_tile_data_copy_destruct);


dague_data_t*
irregular_tile_data_create( dague_data_t **holder,
                            irregular_tiled_matrix_desc_t *desc,
                            dague_data_key_t key, void *ptr,
                            int mb, int nb, size_t elem_size )
{
	dague_data_t *data = *holder;

    if( NULL == data ) {
	    data = OBJ_NEW(dague_data_t);
        data->owner_device = 0;
        data->key = key;
        data->ddesc = (dague_ddesc_t*)desc;
        data->nb_elts = mb*nb*elem_size;

        dague_data_copy_t* data_copy = (dague_data_copy_t*)OBJ_NEW(irregular_tile_data_copy_t);
        dague_data_copy_attach(data, data_copy, 0);
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
            dague_data_copy_t* data_copy = dague_data_copy_new(data, 0);
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

static dague_data_t* get_data(irregular_tiled_matrix_desc_t* desc, int i, int j)
{
	int pos;
	i += desc->i;
	j += desc->j;
	/* Row major column storage */
	pos = (desc->lnt * i) + j;
	assert(0 <= pos && pos < desc->lmt * desc->lnt);
	return desc->local_data_map[pos];
}

static uint32_t irregular_tiled_matrix_rank_of(dague_ddesc_t* d, ...)
{
	int i, j;
	va_list ap;

	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;

	va_start(ap, d);
	i = va_arg(ap, int);
	j = va_arg(ap, int);
	va_end(ap);

	irregular_tile_t* t = get_tile(desc, i, j);

	assert(NULL != t);
	return t->rank;
}

static int32_t irregular_tiled_matrix_vpid_of(dague_ddesc_t* d, ...)
{
	int i, j;
	va_list ap;

	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;

	va_start(ap, d);
	i = va_arg(ap, int);
	j = va_arg(ap, int);
	va_end(ap);

	irregular_tile_t* t = get_tile(desc, i, j);

	assert(NULL != t);
	return t->vpid;
}

static dague_data_t* irregular_tiled_matrix_data_of(dague_ddesc_t* d, ...)
{
	int i, j;
	va_list ap;

	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;

	va_start(ap, d);
	i = va_arg(ap, int);
	j = va_arg(ap, int);
	va_end(ap);

	dague_data_t* t = get_data(desc, i, j);

#if defined(DISTRIBUTED)
	assert(d->myrank == irregular_tiled_matrix_rank_of(d, i, j));
#endif

	/* fprintf(stdout, "data_of (%d;%d), size (%d;%d)\n", i, j, t->mb, t->nb); */

	assert(NULL != t);
	return t;
}

static dague_data_t* irregular_tiled_matrix_C_data_of(dague_ddesc_t* d, ...)
{
	int i, j;
	va_list ap;

	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;

	va_start(ap, d);
	i = va_arg(ap, int);
	j = va_arg(ap, int);
	va_end(ap);

	dague_data_t* t = get_data(desc, i, j);

#if defined(DISTRIBUTED)
	assert(d->myrank == irregular_tiled_matrix_rank_of(d, i, j));
#endif

	assert(NULL != t);
	return t;
}


static void irregular_tiled_matrix_key_to_coords(dague_ddesc_t* d, dague_data_key_t key, int *i, int *j)
{
	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;
	*i = key / desc->lnt;
	*j = key % desc->lnt;
}

static uint32_t irregular_tiled_matrix_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key)
{
	int i, j;
	irregular_tiled_matrix_key_to_coords(ddesc, key, &i, &j);
	return irregular_tiled_matrix_rank_of(ddesc, i, j);
}

static int32_t irregular_tiled_matrix_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key)
{
	int i, j;
	irregular_tiled_matrix_key_to_coords(ddesc, key, &i, &j);
	return irregular_tiled_matrix_vpid_of(ddesc, i, j);
}

static dague_data_t* irregular_tiled_matrix_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key)
{
	int i, j;
	irregular_tiled_matrix_key_to_coords(ddesc, key, &i, &j);
	return irregular_tiled_matrix_data_of(ddesc, i, j);
}

static dague_data_t* irregular_tiled_matrix_C_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key)
{
	int i, j;
	irregular_tiled_matrix_key_to_coords(ddesc, key, &i, &j);

	return irregular_tiled_matrix_C_data_of(ddesc, i, j);
}

static uint32_t irregular_tiled_matrix_data_key(struct dague_ddesc_s *d, ...)
{
	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;
	int i, j;
	va_list ap;

	va_start(ap, d);
	i = va_arg(ap, unsigned int);
	j = va_arg(ap, unsigned int);
	va_end(ap);

	i += desc->i;
	j += desc->j;

	uint32_t k = (i * desc->lnt) + j;

	return k;
}

#if defined(DAGUE_PROF_TRACE)
static int irregular_tiled_matrix_key_to_string(dague_ddesc_t *d, dague_data_key_t key, char * buffer, uint32_t buffer_size)
{
	unsigned int m, n;
	int res;
	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;

	m = key % desc->lnt;
	n = key / desc->lnt;
	res = snprintf(buffer, buffer_size, "(%u, %u)", m, n);
	if (res < 0)
		dague_warning("Wrong key_to_string for tile (%u, %u) key: %u", m, n, key);
	return res;
}
#endif


int tile_is_local(int i, int j, grid_2Dcyclic_t* g)
{
	return (((i/g->strows)%g->rows == g->rrank)&&((j/g->strows)%g->cols == g->crank));
}

int tile_owner(int i, int j, grid_2Dcyclic_t* g)
{
	int rows_tile = g->strows*g->rows;
	int cols_tile = g->stcols*g->cols;

	int iP = i%rows_tile;
	int iQ = j%cols_tile;

	iP /= g->strows;
	iQ /= g->stcols;

	return iP*g->cols+iQ;
}

void irregular_tiled_matrix_desc_init(irregular_tiled_matrix_desc_t* ddesc,
                                      enum tile_coll_type mtype,
                                      unsigned int nodes, unsigned int myrank,
                                      unsigned int lm, unsigned int ln,
                                      unsigned int lmt, unsigned int lnt,
                                      unsigned int* Mtiling, unsigned int* Ntiling,
                                      unsigned int i, unsigned int j,
                                      unsigned int mt, unsigned int nt,
                                      unsigned int P,
                                      void *(*future_resolve_fct)(void*))
{
	unsigned int k;
	dague_ddesc_t *d = (dague_ddesc_t*)ddesc;

	if(nodes < P)
        dague_abort("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d", nodes, P);
    int Q = nodes / P;
    if(nodes != P*Q)
        dague_warning("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d", nodes, P, Q);

	dague_ddesc_init(d, nodes, myrank);

	d->rank_of     = irregular_tiled_matrix_rank_of;
	d->rank_of_key = irregular_tiled_matrix_rank_of_key;
	d->vpid_of     = irregular_tiled_matrix_vpid_of;
	d->vpid_of_key = irregular_tiled_matrix_vpid_of_key;
	d->data_of     = irregular_tiled_matrix_data_of;
	d->data_of_key = irregular_tiled_matrix_data_of_key;
	d->data_key    = irregular_tiled_matrix_data_key;

#if defined(DAGUE_PROF_TRACE)
	d->key_to_string = irregular_tiled_matrix_key_to_string;
#endif

    grid_2Dcyclic_init(&ddesc->grid, myrank, P, Q, 1, 1);

	ddesc->data_map = (irregular_tile_t*)malloc(lmt*lnt*sizeof(irregular_tile_t));

	ddesc->local_data_map = (dague_data_t**)calloc(lmt*lnt, sizeof(dague_data_t*));
	/* lmt, lnt sized arrays */
	ddesc->Mtiling = (unsigned int*)malloc(lmt*sizeof(unsigned int));
	ddesc->Ntiling = (unsigned int*)malloc(lnt*sizeof(unsigned int));

	for (k = 0; k < lmt; ++k) ddesc->Mtiling[k] = Mtiling[k];
	for (k = 0; k < lnt; ++k) ddesc->Ntiling[k] = Ntiling[k];

	ddesc->mtype = mtype;
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
	    if (Ntiling[i] > ddesc->max_mb)
		    ddesc->max_mb = Ntiling[i];
	    for (j = 0; j < lnt; ++j)
		    if (Mtiling[i]*Ntiling[j] > ddesc->max_tile)
			    /* Worst case scenario */
			    ddesc->max_tile = Mtiling[i]*Ntiling[j];
    }

    ddesc->future_resolve_fct = future_resolve_fct;
}

void irregular_tiled_matrix_desc_destroy(irregular_tiled_matrix_desc_t* ddesc)
{
	(void)ddesc;

}

void irregular_tiled_matrix_desc_build(irregular_tiled_matrix_desc_t *ddesc)
{
	(void)ddesc;




}

/* sets up the tile by constructing a new object, then filling specific fields with input parameter */
void irregular_tiled_matrix_desc_set_data(irregular_tiled_matrix_desc_t *ddesc, void *actual_data, int i, int j, int mb, int nb, int vpid, int rank)
{
	(void)actual_data;
	(void)i;
	(void)j;
	(void)nb;
	(void)mb;
	(void)vpid;
	(void)rank;

	uint32_t idx = ((dague_ddesc_t*)ddesc)->data_key((dague_ddesc_t*)ddesc, i, j);

	if (NULL != actual_data) {
		irregular_tile_data_create(
			ddesc->local_data_map+idx, ddesc, idx, actual_data,
			mb, nb, dague_irregular_tiled_matrix_getsizeoftype(ddesc->mtype));

		ddesc->nb_local_tiles++;
	}
	ddesc->data_map[idx].vpid = vpid;
	ddesc->data_map[idx].rank = rank;
}

int
summa_aux_getSUMMALookahead(irregular_tiled_matrix_desc_t *ddesc)
{
    /**
     * Assume that the number of threads per node is constant, and compute the
     * look ahead based on the global information to get the same one on all
     * nodes.
     */
    int nbunits = vpmap_get_nb_total_threads() * ddesc->super.nodes;
    double alpha =  3. * (double)nbunits / ( ddesc->mt * ddesc->nt );

    if ( ddesc->super.nodes == 1 ) {
        /* No look ahaead */
        return summa_imax( ddesc->mt, ddesc->nt );
    }
    else {
        /* Look ahead of at least 2, and that provides 3 tiles per computational units */
        return summa_imax( ceil( alpha ), 2 );
    }
}
