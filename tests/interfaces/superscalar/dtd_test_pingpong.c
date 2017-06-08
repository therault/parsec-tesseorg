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

double time_elapsed;
double sync_time_elapsed;

enum regions {
               TILE_FULL,
             };

int
task_for_timing_0( parsec_execution_unit_t    *context,
                   parsec_execution_context_t *this_task )
{
    (void)context; (void)this_task;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_for_timing_1( parsec_execution_unit_t    *context,
             parsec_execution_context_t *this_task )
{
    (void)context; (void)this_task;

    return PARSEC_HOOK_RETURN_DONE;
}

int
task_rank_0( parsec_execution_unit_t    *context,
             parsec_execution_context_t *this_task )
{
    (void)context;
    int *data;

    parsec_dtd_unpack_args(this_task,
                          UNPACK_DATA,  &data
                          );
    *data *= 2;

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
    *data += 1;

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

    if( world != 2 ) {
        parsec_fatal( "Nope! world is not right, we need exactly two MPI process. "
                      "Try with \"mpirun -np 2 .....\"\n" );
    }

    nb = 1; /* tile_size */
    nt = 2; /* total no. of tiles */
    cores = 1;

    if(argv[1] != NULL){
        cores = atoi(argv[1]);
    }

    parsec = parsec_init( cores, &argc, &argv );

    parsec_handle_t *parsec_dtd_handle = parsec_dtd_handle_new(  );

#if defined(PARSEC_HAVE_MPI)
    parsec_arena_construct(parsec_dtd_arenas[TILE_FULL],
                          nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                          MPI_INT);
#endif

    /* Correctness checking */
    ddescA = create_and_distribute_data(rank, world, nb, nt);
    parsec_ddesc_set_key((parsec_ddesc_t *)ddescA, "A");

    parsec_ddesc_t *A = (parsec_ddesc_t *)ddescA;
    parsec_dtd_ddesc_init(A);

    if( 0 == rank ) {
        parsec_output( 0, "\nChecking correctness of pingpong. We send data from rank 0 to rank 1 "
                      "And vice versa.\nWe start with 0 as data and should end up with 1 after "
                      "the trip.\n\n" );
    }

    parsec_data_copy_t *gdata;
    parsec_data_t *data;
    int *real_data, key;

    if( 0 == rank ) {
        key = A->data_key(A, 0, 0);
        data = A->data_of_key(A, key);
        gdata = data->device_copies[0];
        real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
        *real_data = 0;
        parsec_output( 0, "Node: %d A At key[%d]: %d\n", rank, key, *real_data );
    }

    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( parsec, parsec_dtd_handle );

    parsec_context_start(parsec);

    parsec_insert_task( parsec_dtd_handle, task_rank_0,    0,  "task_rank_0",
                       PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | TILE_FULL | AFFINITY,
                       0 );
    parsec_insert_task( parsec_dtd_handle, task_rank_1,    0,  "task_rank_1",
                       PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | TILE_FULL,
                       PASSED_BY_REF,    TILE_OF_KEY(A, 1), INOUT | TILE_FULL | AFFINITY,
                       0 );

    parsec_dtd_data_flush_all( parsec_dtd_handle, A );

    parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

    parsec_context_wait(parsec);

    if( 0 == rank ) {
        key = A->data_key(A, 0, 0);
        data = A->data_of_key(A, key);
        gdata = data->device_copies[0];
        real_data = PARSEC_DATA_COPY_GET_PTR((parsec_data_copy_t *) gdata);
        parsec_output( 0, "Node: %d A At key[%d]: %d\n", rank, key, *real_data );
        assert( *real_data == 1);
    }

    parsec_arena_destruct(parsec_dtd_arenas[0]);
    parsec_dtd_ddesc_fini( A );
    free_data(ddescA);

    if( 0 == rank ) {
        parsec_output( 0, "\nPingpong is behaving correctly.\n" );
    }

    parsec_handle_free( parsec_dtd_handle );

    /* End of correctness checking */


    /* Start of Pingpong timing */
    if( 0 == rank ) {
        parsec_output( 0, "\nChecking time of pingpong. We send data from rank 0 to rank 1 "
                      "And vice versa.\nWe perform this pingpong for 1000 times and measure the time. "
                      "We report the time for different size of data.\n\n" );
    }

    int repeat_pingpong = 1000;
    int sizes_of_data = 4, i, j;
    int sizes[4] = {100, 1000, 10000, 100000};


    for( i = 0; i < sizes_of_data; i++ ) {
        parsec_dtd_handle = parsec_dtd_handle_new(  );
        parsec_enqueue( parsec, parsec_dtd_handle );
        parsec_context_start(parsec);

        nb = sizes[i];
        nt = 2;

#if defined(PARSEC_HAVE_MPI)
        parsec_arena_construct(parsec_dtd_arenas[TILE_FULL],
                              nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                              MPI_INT);
#endif

        ddescA = create_and_distribute_data(rank, world, nb, nt);
        parsec_ddesc_set_key((parsec_ddesc_t *)ddescA, "A");

        parsec_ddesc_t *A = (parsec_ddesc_t *)ddescA;
        parsec_dtd_ddesc_init(A);

        SYNC_TIME_START();

        for( j = 0; j < repeat_pingpong; j++ ) {
            parsec_insert_task( parsec_dtd_handle, task_rank_0,    0,  "task_for_timing_0",
                               PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | TILE_FULL | AFFINITY,
                               0 );
            parsec_insert_task( parsec_dtd_handle, task_rank_1,    0,  "task_for_timing_1",
                               PASSED_BY_REF,    TILE_OF_KEY(A, 0), INOUT | TILE_FULL,
                               PASSED_BY_REF,    TILE_OF_KEY(A, 1), INOUT | TILE_FULL | AFFINITY,
                               0 );
        }

        parsec_dtd_data_flush_all( parsec_dtd_handle, A );
        /* finishing all the tasks inserted, but not finishing the handle */
        parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

        parsec_context_wait(parsec);
        SYNC_TIME_PRINT(rank, ("\tSize of message : %ld bytes\tTime for each pingpong : %12.5f\n", sizes[i]*sizeof(int), sync_time_elapsed/repeat_pingpong));

        parsec_arena_destruct(parsec_dtd_arenas[0]);
        parsec_dtd_ddesc_fini( A );
        free_data(ddescA);
    }

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}
