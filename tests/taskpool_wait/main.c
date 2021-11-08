#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "ptg_tp.h"
#include "dtd_tp.h"

int main(int argc, char *argv[]) {
    int provided, err, world_size, my_rank;
    parsec_taskpool_t *ptg_tp1, *ptg_tp2;
    parsec_taskpool_t *dtd_tp1, *dtd_tp2;
    two_dim_block_cyclic_t A;
    enum matrix_type x;
    int nb = 1;
    int rc;

    err = 0;

    parsec_context_t *parsec;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    two_dim_block_cyclic_init(&A, matrix_Integer, matrix_Tile, world_size, my_rank, 1, 1, world_size, world_size, 0, 0, world_size, world_size, 1, 1, 1);
    parsec_data_collection_set_key(&A.super.super, "A");

    parsec_matrix_add2arena_rect( &parsec_dtd_arenas_datatypes[0],
                                  parsec_datatype_int32_t,
                                  nb, 1, nb );

    /* Registering the dtd_handle with PARSEC context */
    rc = parsec_context_add_taskpool( parsec, dtd_tp1 );
    PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");

    new_dtd_taskpool(dtd_tp1, &A, 1);

    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_taskpool_free( dtd_tp1 );

    parsec_fini(&parsec);
    MPI_Finalize();
    return err;
}