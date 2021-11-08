#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "ptg_tp.h"
#include "dtd_tp.h"

static int TILE_FULL;

int main(int argc, char *argv[]) {
    int provided, err, world_size, my_rank;
    parsec_taskpool_t *ptg_tp1, *ptg_tp2;
    parsec_taskpool_t *dtd_tp1, *dtd_tp2;
    parsec_arena_datatype_t *adt;
    two_dim_block_cyclic_t A;
    int nb = 4;
    int rc;

    err = 0;

    parsec_context_t *parsec;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    parsec = parsec_init(-1, NULL, NULL);

    two_dim_block_cyclic_init(&A, matrix_Integer, matrix_Tile, world_size, my_rank, 1, 1, nb*world_size, nb*world_size, 0, 0, nb*world_size, nb*world_size, 1, 1, world_size);
    parsec_data_collection_set_key(&A.super.super, "A");
    parsec_dtd_data_collection_init(&A.super.super);

    adt = parsec_dtd_create_arena_datatype(parsec, &TILE_FULL);
    parsec_matrix_add2arena_rect( adt,
                                  parsec_datatype_int32_t,
                                  nb, 1, nb );

    printf("Single PTG taskpool, waited with parsec_context_wait\n");
    ptg_tp1 = parsec_ptg_tp_new(&A, 1);
    parsec_enqueue(parsec, ptg_tp1);
    parsec_context_start(parsec);
    parsec_context_wait(parsec);
    parsec_taskpool_free(ptg_tp1);

    printf("Single PTG taskpool, waited with parsec_taskpool_wait then the context is put to sleep with parsec_context_wait\n");
    ptg_tp1 = parsec_ptg_tp_new(&A, 1);
    parsec_enqueue(parsec, ptg_tp1);
    parsec_context_start(parsec);
    parsec_taskpool_wait(ptg_tp1);
    parsec_context_wait(parsec);
    parsec_taskpool_free(ptg_tp1);

    printf("Two PTG taskpools, both waited with parsec_context_wait\n");
    ptg_tp1 = parsec_ptg_tp_new(&A, 1);
    ptg_tp2 = parsec_ptg_tp_new(&A, 5);
    parsec_enqueue(parsec, ptg_tp1);
    parsec_enqueue(parsec, ptg_tp2);
    parsec_context_start(parsec);
    parsec_context_wait(parsec);
    parsec_taskpool_free(ptg_tp1);
    parsec_taskpool_free(ptg_tp2);

    printf("Two PTG taskpools, waited (in reverse order of completion) with parsec_taskpool_wait then the context is waited upon\n");
    ptg_tp1 = parsec_ptg_tp_new(&A, 1);
    ptg_tp2 = parsec_ptg_tp_new(&A, 5);
    parsec_enqueue(parsec, ptg_tp1);
    parsec_enqueue(parsec, ptg_tp2);
    parsec_context_start(parsec);
    parsec_taskpool_wait(ptg_tp2);
    parsec_taskpool_wait(ptg_tp1);
    parsec_context_wait(parsec);
    parsec_taskpool_free(ptg_tp1);
    parsec_taskpool_free(ptg_tp2);

    dtd_tp1 = parsec_dtd_taskpool_new();

    printf("Single DTD taskpool, waited with parsec_context_wait\n");
    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp1 );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    new_dtd_taskpool(dtd_tp1, TILE_FULL, &A, 1);

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free( dtd_tp1 );

    printf("All tests done, cleaning up data\n");

    parsec_matrix_del2arena(adt);
    PARSEC_OBJ_RELEASE(adt->arena);
    parsec_dtd_destroy_arena_datatype(parsec, TILE_FULL);
    parsec_dtd_data_collection_fini(&A.super.super);

    parsec_fini(&parsec);
    MPI_Finalize();
    return err;
}