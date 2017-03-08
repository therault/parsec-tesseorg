
/*
 * Copyright (c) 2016-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec/devices/device.h"
#include "parsec/parsec_internal.h"
#include "parsec/debug.h"
#include "irregular_tiled_matrix.h"
#include "irregular_subtile.h"
#include "parsec/vpmap.h"
#include "parsec.h"
#include "parsec/data.h"
#include "dplasma.h"

#include <math.h>

#ifdef PARSEC_HAVE_MPI
#include <mpi.h>
#endif /* PARSEC_HAVE_MPI */

static uint32_t       irregular_tiled_matrix_rank_of(     parsec_ddesc_t* ddesc, ...);
static uint32_t       irregular_tiled_matrix_rank_of_key( parsec_ddesc_t* ddesc, parsec_data_key_t key);
static int32_t        irregular_tiled_matrix_vpid_of(     parsec_ddesc_t* ddesc, ...);
static int32_t        irregular_tiled_matrix_vpid_of_key( parsec_ddesc_t* ddesc, parsec_data_key_t key);
static parsec_data_t* irregular_tiled_matrix_data_of(     parsec_ddesc_t* ddesc, ...);
static parsec_data_t* irregular_tiled_matrix_data_of_key( parsec_ddesc_t* ddesc, parsec_data_key_t key);
static uint32_t       irregular_tiled_matrix_coord_to_key(parsec_ddesc_t *ddesc, ...);
static void           irregular_tiled_matrix_key_to_coord(parsec_ddesc_t *ddesc, parsec_data_key_t key, int *i, int *j);

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
        data->ddesc = (parsec_ddesc_t*)desc;
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

static uint32_t irregular_tiled_matrix_rank_of(parsec_ddesc_t* d, ...)
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

static int32_t irregular_tiled_matrix_vpid_of(parsec_ddesc_t* d, ...)
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

static parsec_data_t* irregular_tiled_matrix_data_of(parsec_ddesc_t* d, ...)
{
    int i, j;
    va_list ap;

    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;

    va_start(ap, d);
    i = va_arg(ap, int);
    j = va_arg(ap, int);
    va_end(ap);

#if defined(DISTRIBUTED)
    assert(d->myrank == irregular_tiled_matrix_rank_of(d, i, j));
#endif

    parsec_data_t* t = get_data(desc, i, j);
    assert(NULL != t);
    return t;
}

static void irregular_tiled_matrix_key_to_coord(parsec_ddesc_t* d, parsec_data_key_t key, int *i, int *j)
{
    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;
    *i = key / desc->lnt;
    *j = key % desc->lnt;
}

static uint32_t irregular_tiled_matrix_rank_of_key(parsec_ddesc_t* ddesc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(ddesc, key, &i, &j);
    return irregular_tiled_matrix_rank_of(ddesc, i, j);
}

static int32_t irregular_tiled_matrix_vpid_of_key(parsec_ddesc_t* ddesc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(ddesc, key, &i, &j);
    return irregular_tiled_matrix_vpid_of(ddesc, i, j);
}

static parsec_data_t* irregular_tiled_matrix_data_of_key(parsec_ddesc_t* ddesc, parsec_data_key_t key)
{
    int i, j;
    irregular_tiled_matrix_key_to_coord(ddesc, key, &i, &j);
    return irregular_tiled_matrix_data_of(ddesc, i, j);
}

static uint32_t irregular_tiled_matrix_coord_to_key(struct parsec_ddesc_s *d, ...)
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

#if defined(PARSEC_PROF_TRACE)
static int irregular_tiled_matrix_key_to_string(parsec_ddesc_t *d, parsec_data_key_t key, char * buffer, uint32_t buffer_size)
{
    unsigned int m, n;
    int res;
    irregular_tiled_matrix_desc_t* desc = (irregular_tiled_matrix_desc_t*)d;

    m = key % desc->lnt;
    n = key / desc->lnt;
    res = snprintf(buffer, buffer_size, "(%u, %u)", m, n);
    if (res < 0)
        parsec_warning("Wrong key_to_string for tile (%u, %u) key: %u", m, n, key);
    return res;
}
#endif


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
    int res = desc->Mtiling[m] * desc->Ntiling[n];
    /* fprintf(stdout, "m = %d, n = %d > res = %d\n", m, n, res); */
    assert(0 < res && res <= desc->m * desc->n);
    return res;
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
    parsec_ddesc_t *d = (parsec_ddesc_t*)ddesc;

    if(nodes < P)
        parsec_fatal("Block Cyclic Distribution:\tThere are not enough nodes (%d) to make a process grid with P=%d", nodes, P);

    int Q = nodes / P;
    if(nodes != P*Q)
        parsec_warning("Block Cyclic Distribution:\tNumber of nodes %d doesn't match the process grid %dx%d", nodes, P, Q);

    parsec_ddesc_init(d, nodes, myrank);

    d->rank_of     = irregular_tiled_matrix_rank_of;
    d->rank_of_key = irregular_tiled_matrix_rank_of_key;
    d->vpid_of     = irregular_tiled_matrix_vpid_of;
    d->vpid_of_key = irregular_tiled_matrix_vpid_of_key;
    d->data_of     = irregular_tiled_matrix_data_of;
    d->data_of_key = irregular_tiled_matrix_data_of_key;
    d->data_key    = irregular_tiled_matrix_coord_to_key;

#if defined(PARSEC_PROF_TRACE)
    d->key_to_string = irregular_tiled_matrix_key_to_string;
#endif

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
}

void irregular_tiled_matrix_destroy_data(irregular_tiled_matrix_desc_t* ddesc)
{
    int i, j;
    for (i = 0; i < ddesc->lmt; ++i)
        for (j = 0; j < ddesc->lnt; ++j)
            if (ddesc->super.myrank == irregular_tiled_matrix_rank_of(&ddesc->super, i, j)) {
                uint32_t idx = ((parsec_ddesc_t*)ddesc)->data_key((parsec_ddesc_t*)ddesc, i, j);
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

    parsec_ddesc_destroy((parsec_ddesc_t*)ddesc);
}

/* sets up the tile by constructing a new object, then filling specific fields with input parameter */
 void irregular_tiled_matrix_desc_set_data(irregular_tiled_matrix_desc_t *ddesc, void *actual_data, uint32_t idx, int mb, int nb, int vpid, int rank)
{
    (void)actual_data;
    (void)idx;
    (void)vpid;
    (void)rank;

    if (NULL != actual_data) {
        parsec_data_create(ddesc->local_data_map+idx,(parsec_ddesc_t*)ddesc, idx, actual_data,
                           mb*nb*parsec_irregular_tiled_matrix_getsizeoftype(ddesc->mtype));
        /* Workaround: irregular_tile_data_create( */
        /*     ddesc->local_data_map+idx, ddesc, idx, actual_data, */
        /*     mb, nb, parsec_irregular_tiled_matrix_getsizeoftype(ddesc->mtype)); */

        ddesc->nb_local_tiles++;
    }
    ddesc->data_map[idx].vpid = vpid;
    ddesc->data_map[idx].rank = rank;

    assert (rank == ((parsec_ddesc_t*)ddesc)->myrank || actual_data == NULL);
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
    double alpha =  9. * (double)nbunits / ( ddesc->mt * ddesc->nt );

    if ( ddesc->super.nodes == 1 ) {
        /* No look ahaead */
        return summa_imax( ddesc->mt, ddesc->nt );
    }
    else {
        /* Look ahead of at least 2, and that provides 3 tiles per computational units */
        return summa_imax( ceil( alpha ), 2 );
    }
}

void
export_pythons(irregular_tiled_matrix_desc_t *A, irregular_tiled_matrix_desc_t *B, irregular_tiled_matrix_desc_t *C, int P, int Q, int N, int NB, int MB, int psize)
{
    int i, j, k, min_dgemm, max_dgemm;
    unsigned int *Mtiling = C->Mtiling;
    unsigned int *Ntiling = C->Ntiling;
    unsigned int *Ktiling = A->Ntiling;
    int MT = C->lmt;
    int NT = C->lnt;
    int KT = A->lnt;
    (void)B;
    
    min_dgemm = max_dgemm = Mtiling[0]*Ntiling[0]*Ktiling[0]*2;

    for (i = 0; i < MT; ++i)
        for (j = 0; j < NT; ++j)
            for (k = 0; k < KT; ++k) {
                int dgemm = Mtiling[i]*Ntiling[j]*Ktiling[k]*2;
                if (dgemm < min_dgemm) min_dgemm = dgemm;
                if (dgemm > max_dgemm) max_dgemm = dgemm;
            }

    int mod = max_dgemm / 100;
    float *range = (float*)malloc(101*sizeof(float));
    int *freqs = (int*)malloc(101*sizeof(int));

    float *drange = (float*)malloc(101*psize*sizeof(float));
    int *distrib = (int*)malloc(101*psize*sizeof(int));
    
    for (i = 0; i <= 100; ++i) range[i] = i*mod/1000000000.;
    for (i = 0; i <= 100; ++i) freqs[i] = 0;

    for (i = 0; i < 101; ++i) for (j = 0; j < psize; ++j) drange[i*psize+j] = (1.*i+j/(psize+1.))*mod/1000000000.;
    for (i = 0; i < 101*psize; ++i) distrib[i] = 0;

    for (i = 0; i < MT; ++i)
        for (j = 0; j < NT; ++j) {
            uint32_t proc = C->data_map[(C->lnt * i) + j].rank;
            for (k = 0; k < KT; ++k) {
                int dgemm = Mtiling[i]*Ntiling[j]*Ktiling[k]*2/mod;
                freqs[dgemm]++;
                distrib[101*proc+dgemm]++;
            }
        }

    /* Export as python histo plot */
    char buf[1024];
    sprintf(buf, "flops_distribution_N%d_t%d.py", N, NB);
    FILE *fp = fopen(buf, "w+");

    fprintf(fp, "import numpy as np\n");
    fprintf(fp, "import matplotlib.pyplot as plt\n\n");
    
    fprintf(fp, "x = [");
    for (i = 0; i <= 100; ++i) fprintf(fp, "%s%lf", (i==0)?"":", ", range[i]);
    fprintf(fp, "]\n\n");

    fprintf(fp, "y = [");
    for (i = 0; i <= 100; ++i) fprintf(fp, "%s%d", (i==0)?"":", ", freqs[i]);
    fprintf(fp, "]\n\n");

    fprintf(fp, "delta = 0.5*(x[1]-x[0])\n");
    fprintf(fp, "plt.bar(x, y, delta)\n");
    fprintf(fp, "plt.axvline(x=%lf, color='red')\n", 2.*MB*MB*MB/1000000000);
    fprintf(fp, "plt.show()\n");
    fclose(fp);

    /* Export flops distrib*/
    if (psize > 1) {
        sprintf(buf, "flops_distribution_N%d_t%d_np%d.py", N, NB, psize);
        fp = fopen(buf, "w+");

        fprintf(fp, "import numpy as np\n");
        fprintf(fp, "import matplotlib.pyplot as plt\n\n");

        fprintf(fp, "f, ax = plt.subplots(%d, %d, sharex='col', sharey='row', figsize=(%d, %d))\n\n", P, Q, 5*Q, 3*P);

        fprintf(fp, "x = [");
        for (i = 0; i <= 100; ++i) fprintf(fp, "%s%lf", (i==0)?"":", ", range[i]);
        fprintf(fp, "]\n\n");
        fprintf(fp, "delta = x[1]-x[0]\n");

        int p, q;
        for (p = 0; p < psize; ++p) {
            fprintf(fp, "y%d = [", p);
            for (i = 0; i < 101; ++i) fprintf(fp, "%s%d", (i==0)?"":", ", distrib[101*p+i]);
            fprintf(fp, "]\n\n");
        }

        for (q = 0; q < Q; ++q)
            for (p = 0; p < P; ++p) {
                fprintf(fp, "ax[%d,%d].bar(x, y%d, delta)\n", p, q, q*P+p);
                fprintf(fp, "ax[%d,%d].axvline(x=%lf, color='red')\n", p, q, 2.*MB*MB*MB/1000000000);
            }
        fprintf(fp, "\nplt.suptitle('Number of occurences against tile size as Flops, N=%d, t=%d, PxQ=(%dx%d)\nRed line is the nominal %d tile size.')\n", N, NB, P, Q, NB);
        fprintf(fp, "plt.show()\n");
        fclose(fp);
    }

    free(range);
    free(freqs);
    free(drange);
    free(distrib);
}
