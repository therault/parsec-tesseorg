#include "parsec/parsec_config.h"

/* system and io */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/* parsec things */
#include "parsec.h"
#include "parsec/profiling.h"
#ifdef PARSEC_VTRACE
#include "parsec/vt_user.h"
#endif

#include "common_data.h"
#include "common_timing.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

enum regions {
               TILE_FULL,
             };

int
task_rank_0( parsec_execution_unit_t    *context,
             parsec_execution_context_t *this_task )
{
    (void)context;
    int *data;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data
                          );

    if(this_task->parsec_handle->context->my_rank == 5)sleep(1);

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_1( parsec_execution_unit_t    *context,
             parsec_execution_context_t *this_task )
{
    (void)context;
    int *data;
    int *second_data;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data,
                          UNPACK_DATA,  &second_data
                          );

    *second_data += *data;
    printf( "My rank: %d, diff: %d\n", this_task->parsec_handle->context->my_rank, *data );

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char **argv)
{
    parsec_context_t* parsec;
    int rank, world, cores;
    int nb, nt;
    tiled_matrix_desc_t *ddescA;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    nb = 1; /* tile_size */
    nt = world; /* total no. of tiles */
    cores = 20;

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    parsec = parsec_init( cores, &argc, &argv );

    parsec_handle_t *parsec_dtd_handle = parsec_dtd_handle_new(  );

#if defined(PARSEC_HAVE_MPI)
    parsec_arena_construct(parsec_dtd_arenas[0],
                          nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                          MPI_INT);
#endif

    /* Correctness checking */
    ddescA = create_and_distribute_data(rank, world, nb, nt);
    parsec_ddesc_set_key((parsec_ddesc_t *)ddescA, "A");

    parsec_ddesc_t *A = (parsec_ddesc_t *)ddescA;
    parsec_dtd_ddesc_init(A);

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data, key;
    int root = 0, i;

    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( parsec, parsec_dtd_handle );

    parsec_context_start(parsec);

// *********************
    if( rank == root) {
        printf("Root: %d\n\n", root );
    }

    key = A->data_key(A, rank, 0);
    data = A->data_of_key(A, key);
    gdata = data->device_copies[0];
    real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
    *real_data = rank;

    for( i = 0; i < world; i ++ ) {
        if( root != i ) {
            parsec_insert_task( parsec_dtd_handle, task_rank_0,    0,  "task_rank_0",
                                PASSED_BY_REF,    TILE_OF_KEY(A, i), INOUT | TILE_FULL | AFFINITY,
                                0 );

            parsec_insert_task( parsec_dtd_handle, task_rank_1,    0,  "task_rank_0",
                                PASSED_BY_REF,    TILE_OF_KEY(A, i),    INOUT | TILE_FULL,
                                PASSED_BY_REF,    TILE_OF_KEY(A, root), INOUT | TILE_FULL | AFFINITY,
                                0 );
        }
    }
//******************
    parsec_dtd_data_flush_all( parsec_dtd_handle, A );

    parsec_dtd_handle_wait( parsec, parsec_dtd_handle );
    parsec_context_wait(parsec);

    parsec_handle_free( parsec_dtd_handle );

    parsec_arena_destruct(parsec_dtd_arenas[0]);
    parsec_dtd_ddesc_fini( A );
    free_data(ddescA);

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
