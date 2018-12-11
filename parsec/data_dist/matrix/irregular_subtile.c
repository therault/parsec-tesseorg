/*
 * Copyright (c) 2017 The University of Tennessee and The University
 *                    of Tennessee Research Foundation.  All rights
 *                    reserved.
 */

#include "parsec/data_dist/matrix/irregular_tiled_matrix.h"
#include "parsec/data_dist/matrix/irregular_subtile.h"
#include "parsec/vpmap.h"

static uint32_t       irregular_subtile_rank_of(     parsec_data_collection_t* dc, ...);
static uint32_t       irregular_subtile_rank_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static int32_t        irregular_subtile_vpid_of(     parsec_data_collection_t* dc, ...);
static int32_t        irregular_subtile_vpid_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static parsec_data_t* irregular_subtile_data_of(     parsec_data_collection_t* dc, ...);
static parsec_data_t* irregular_subtile_data_of_key( parsec_data_collection_t* dc, parsec_data_key_t key);
static void           irregular_subtile_key_to_coord(parsec_data_collection_t *desc, parsec_data_key_t key, int *m, int *n);

two_dim_block_cyclic_t* recursive_fake_Cdist(const two_dim_block_cyclic_t* original)
{
    two_dim_block_cyclic_t *copy = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    two_dim_block_cyclic_init(copy, matrix_RealDouble, matrix_Tile,
                              original->super.super.nodes, original->super.super.myrank,
                              1, 1, original->super.lm, original->super.ln,
                              0, 0, original->super.lm, original->super.ln,
                              1, 1, original->grid.rows);

    copy->super.super.data_of     = NULL;
    copy->super.super.data_of_key = NULL;
    copy->super.super.rank_of     = irregular_subtile_rank_of;
    copy->super.super.rank_of_key = irregular_subtile_rank_of_key;
    copy->super.super.vpid_of     = irregular_subtile_vpid_of;
    copy->super.super.vpid_of_key = irregular_subtile_vpid_of_key;

    return copy;
}

irregular_subtile_desc_t *irregular_subtile_desc_create(const irregular_tiled_matrix_desc_t *tdesc,
                                                        int mm, int nn, /* Tile in tdesc */
                                                        int mt, int nt)
{
    int i, j;
    irregular_subtile_desc_t *sdesc = malloc(sizeof(irregular_subtile_desc_t));
    parsec_data_collection_t *o = &(sdesc->super.super);

    int m = tdesc->Mtiling[mm];
    int n = tdesc->Ntiling[nn];
    int MT = mt; /* 1+(m-1)/opttile; */
    int NT = nt; /* 1+(n-1)/opttile; */

    unsigned int *Mtiling = (unsigned int*)malloc(MT*sizeof(unsigned int));
    unsigned int *Ntiling = (unsigned int*)malloc(NT*sizeof(unsigned int));

    int mshare = m/MT;
    int nshare = n/NT;
    for (i = 0; i < MT; ++i) Mtiling[i] = (i < MT-1) ? mshare : m - (MT-1)*mshare;
    for (i = 0; i < NT; ++i) Ntiling[i] = (i < NT-1) ? nshare : n - (NT-1)*nshare;
    
    /* Initialize the tiled_matrix descriptor */
    irregular_tiled_matrix_desc_init( &(sdesc->super), tdesc->mtype,
                                      tdesc->super.nodes, tdesc->super.myrank,
                                      m, n,
                                      MT, NT,
                                      Mtiling, Ntiling,
                                      0, 0, MT, NT,
                                      tdesc->grid.rows, NULL);

    sdesc->super.storage = matrix_Lapack;
    sdesc->super.llm = m;
    sdesc->super.lln = n;
    sdesc->super.nb_local_tiles = MT * NT;
    sdesc->super.local_data_map = (parsec_data_t**)calloc(sdesc->super.nb_local_tiles, sizeof(parsec_data_t*));

    /* Fetch the data from dc and coordinates */
    parsec_data_t *data = ((parsec_data_collection_t*)tdesc)->data_of((parsec_data_collection_t*)tdesc, mm, nn);
    /* Fetch the last version data_copy of this data */
    parsec_data_copy_t* data_copy = parsec_data_get_copy(data, 0);
    /* Access the actual pointer */
    void *actual_data = data_copy->device_private;

    for (i = 0; i < MT; ++i)
        for (j = 0; j < NT; ++j) {
            int offset = (i * mshare * n + j * nshare) * parsec_datadist_getsizeoftype(tdesc->mtype);
            uint32_t idx = ((parsec_data_collection_t*)sdesc)->data_key((parsec_data_collection_t*)sdesc, i, j);

            parsec_data_create(sdesc->super.local_data_map + idx,
                               (parsec_data_collection_t*)sdesc,
                               idx, ((char*)actual_data) + offset,
                               Mtiling[i] * Ntiling[j] * parsec_datadist_getsizeoftype(tdesc->mtype));
        }

    sdesc->mat = NULL;  /* No data associated with the matrix yet */
    //sdesc->mat  = tdesc->super.data_of( (parsec_data_collection_t*)tdesc, mt, nt );
    sdesc->vpid = 0;

    /* set the methods */
    o->rank_of      = irregular_subtile_rank_of;
    o->vpid_of      = irregular_subtile_vpid_of;
    o->data_of      = irregular_subtile_data_of;
    o->rank_of_key  = irregular_subtile_rank_of_key;
    o->vpid_of_key  = irregular_subtile_vpid_of_key;
    o->data_of_key  = irregular_subtile_data_of_key;

    /* Memory is allready registered at direct upper level */
    o->register_memory   = NULL;
    o->unregister_memory = NULL;

    return sdesc;
}

static uint32_t irregular_subtile_rank_of(parsec_data_collection_t* dc, ...)
{
    return dc->myrank;
}

static uint32_t irregular_subtile_rank_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    (void)key;
    return dc->myrank;
}

static int32_t irregular_subtile_vpid_of(parsec_data_collection_t* dc, ...)
{
    int pq = vpmap_get_nb_vp();
    if ( pq == 1 )
        return 0;

    int i, j, vpid = 0;
    (void)i;
    (void)j;
    va_list ap;
    /* Get coordinates */
    va_start(ap, dc);
    i = (int)va_arg(ap, unsigned int);
    j = (int)va_arg(ap, unsigned int);
    va_end(ap);

    return vpid;
}

static int32_t irregular_subtile_vpid_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    int m, n;
    irregular_subtile_key_to_coord(dc, key, &m, &n);
    return irregular_subtile_vpid_of(dc, key);
}

static parsec_data_t* irregular_subtile_data_of(parsec_data_collection_t* dc, ...)
{
    int i, j;
    va_list ap;
    irregular_subtile_desc_t * sdesc = (irregular_subtile_desc_t *)dc;

    /* Get coordinates */
    va_start(ap, dc);
    i = (int)va_arg(ap, unsigned int);
    j = (int)va_arg(ap, unsigned int);
    va_end(ap);

    parsec_data_t* t = sdesc->super.local_data_map[sdesc->super.lnt * i + j];
    assert(NULL != t);
    return t;
}

static parsec_data_t* irregular_subtile_data_of_key(parsec_data_collection_t* dc, parsec_data_key_t key)
{
    int m, n;
    irregular_subtile_key_to_coord(dc, key, &m, &n);
    return irregular_subtile_data_of(dc, m, n);
}

static void irregular_subtile_key_to_coord(parsec_data_collection_t *dc, parsec_data_key_t key, int *m, int *n)
{
    irregular_tiled_matrix_desc_t *tdesc = (irregular_tiled_matrix_desc_t *)dc;

    *m = key % tdesc->lmt;
    *n = key / tdesc->lmt;
}
