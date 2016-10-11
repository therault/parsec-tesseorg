
/*
 * Copyright (c) 2016      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
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


static uint32_t      irregular_tiled_matrix_rank_of(    dague_ddesc_t* ddesc, ...);
static uint32_t      irregular_tiled_matrix_rank_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);

static int32_t       irregular_tiled_matrix_vpid_of(    dague_ddesc_t* ddesc, ...);
static int32_t       irregular_tiled_matrix_vpid_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);

static dague_data_t* irregular_tiled_matrix_data_of(    dague_ddesc_t* ddesc, ...);
static dague_data_t* irregular_tiled_matrix_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);
static dague_data_t* irregular_tiled_matrix_C_data_of(    dague_ddesc_t* ddesc, ...);
static dague_data_t* irregular_tiled_matrix_C_data_of_key(dague_ddesc_t* ddesc, dague_data_key_t key);

static uint32_t      irregular_tiled_matrix_data_key(   dague_ddesc_t *ddesc, ...);


/* dague_data_t* irregular_tile_data_new(void) */
/* { */
/* 	return (dague_data_t*)OBJ_NEW(irregular_tile_data_t); */
/* } */



static irregular_tile_data_t* get_tile(irregular_tiled_matrix_desc_t* desc, int i, int j)
{
	int pos;
	i += desc->i;
	j += desc->j;
	/* Row major column storage */
	pos = (desc->lnt * i) + j;
	assert(0 <= pos && pos < desc->lmt * desc->lnt);
	return desc->data_map[pos];
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

	irregular_tile_data_t* t = get_tile(desc, i, j);

	if (NULL != t)
		return t->rank;
	return (uint32_t)-1;
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

	irregular_tile_data_t* t = get_tile(desc, i, j);

	if (NULL != t)
		return t->vpid;
	return (uint32_t)-1;
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

	irregular_tile_data_t* t = get_tile(desc, i, j);

#if defined(DISTRIBUTED)
	assert(d->myrank == irregular_tiled_matrix_rank_of(d, i, j));
#endif

	fprintf(stdout, "data_of (%d;%d), size (%d;%d)\n", i, j, t->mb, t->nb);

	if (NULL != t)
		return t->data;
	return (dague_data_t*)NULL;
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

	irregular_tile_data_t* t = get_tile(desc, i, j);

#if defined(DISTRIBUTED)
	assert(d->myrank == irregular_tiled_matrix_rank_of(d, i, j));
#endif

	if (NULL != t)
		return t->data;
	return (dague_data_t*)NULL;
}

static void* irregular_tiled_matrix_allocate_if_null(dague_ddesc_t *d, int i, int j)
{
	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;
	irregular_tile_data_t* t = get_tile(desc, i, j);

	if (NULL == t->data) {
		/* void *buf = malloc(sizeof()*desc->itiling[i]*desc->jtiling[j]); */


	}
	return NULL;
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
	irregular_tiled_matrix_allocate_if_null(ddesc, i, j);
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

	fprintf(stdout, "Generating key for tile (%d;%d) -> %d\n",i,j,k);
	return k;
}

#if defined(DAGUE_PROF_TRACE)
static int irregular_tiled_matrix_key_to_string(dague_ddesc_t *d, dague_data_key_t key, char * buffer, uint32_t buffer_size)
{
	(void)key;
	(void)buffer;
	(void)buffer_size;

	irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;

	(void)desc;

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
                                      unsigned int* itiling, unsigned int* jtiling,
                                      unsigned int i, unsigned int j,
                                      unsigned int mt, unsigned int nt,
                                      unsigned int P)
{
	dague_ddesc_t *d = (dague_ddesc_t*)ddesc;
	dague_ddesc_init(d, nodes, myrank);

    if(nodes < P)
        dague_abort("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d", nodes, P);
    int Q = nodes / P;
    if(nodes != P*Q)
        dague_warning("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d", nodes, P, Q);

    grid_2Dcyclic_init(&ddesc->grid, myrank, P, Q, 1, 1);

	d->rank_of     = irregular_tiled_matrix_rank_of;
	d->rank_of_key = irregular_tiled_matrix_rank_of_key;
	d->vpid_of     = irregular_tiled_matrix_vpid_of;
	d->vpid_of_key = irregular_tiled_matrix_vpid_of_key;
	d->data_of     = irregular_tiled_matrix_data_of;
	d->data_of_key = irregular_tiled_matrix_data_of_key;
	d->data_key    = irregular_tiled_matrix_data_key;

#if defined(DAGUE_PROF_TRACE)
	d->key_to_string = tiled_matrix_key_to_string;
#endif

	ddesc->dtype = irregular_tiled_matrix_desc_type;
	ddesc->mtype = mtype;
	ddesc->lm = lm;
	ddesc->ln = ln;
	ddesc->lmt = lmt;
	ddesc->lnt = lnt;
	ddesc->mt = mt;
	ddesc->nt = nt;
	/* lmt, lnt sized arrays */
	ddesc->itiling = (unsigned int*)malloc(lmt*sizeof(unsigned int));
	ddesc->jtiling = (unsigned int*)malloc(lnt*sizeof(unsigned int));
	unsigned int k;
	for (k = 0; k < lmt; ++k) ddesc->itiling[k] = itiling[k];
	for (k = 0; k < lnt; ++k) ddesc->jtiling[k] = jtiling[k];
	ddesc->i = i;
	ddesc->j = j;

	ddesc->data_map = (irregular_tile_data_t**)malloc(lmt*lnt*sizeof(irregular_tile_data_t*));
	for (i = 0; i < lmt; ++i) {
		for (j = 0; j < lnt; ++j) {
			int idx = i*lnt+j;
			irregular_tile_data_t *t = (irregular_tile_data_t*)malloc(sizeof(irregular_tile_data_t));
#if defined(DAGUE_DEBUG)
			t->magic = LET_THE_MAGIC_HAPPENS;
#endif
			t->rank = tile_owner(i, j, &(ddesc->grid));
			t->vpid = 0;
			t->tileld = -1;
			t->mb = -1;
			t->nb = -1;
			t->data = NULL;
			ddesc->data_map[idx] = t;
		}
	}
	ddesc->nb_local_tiles = 0;
}

void irregular_tiled_matrix_desc_destroy(irregular_tiled_matrix_desc_t* ddesc)
{
	(void)ddesc;

}

void irregular_tiled_matrix_desc_build(irregular_tiled_matrix_desc_t *ddesc)
{
	(void)ddesc;




}


void irregular_tiled_matrix_desc_set_data(irregular_tiled_matrix_desc_t *ddesc, void *actual_data, int i, int j, int mb, int nb, int vpid, int rank)
{
	(void)actual_data;
	(void)i;
	(void)j;
	(void)nb;
	(void)mb;
	(void)vpid;
	(void)rank;

	int idx = (ddesc->i+i)*ddesc->lnt+(ddesc->j+j);
	irregular_tile_data_t *t = ddesc->data_map[idx];

	if (tile_is_local(i, j, &(ddesc->grid))) {
		ddesc->itiling[ddesc->i+i] = mb;
		ddesc->jtiling[ddesc->j+j] = nb;
#if defined(DAGUE_DEBUG)
		t->magic = LET_THE_MAGIC_HAPPENS;
#endif
		t->super.key = ddesc->super.data_key((dague_ddesc_t*)ddesc, i, j);
		t->rank = rank;
		t->vpid = vpid;
		t->data = actual_data;
		dague_data_create((dague_data_t**)ddesc->data_map+idx, (dague_ddesc_t*)ddesc,
		                  (dague_data_key_t)(idx), actual_data, nb*mb);
	}
	ddesc->nb_local_tiles++;
}

int
summa_aux_getSUMMALookahead( irregular_tiled_matrix_desc_t *ddesc )
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
