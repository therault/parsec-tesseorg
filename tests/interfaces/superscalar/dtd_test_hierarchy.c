#include "parsec_config.h"

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

#include "common_timing.h"
#include "common_data.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif  /* defined(PARSEC_HAVE_MPI) */

/* This testing shows graph pruning as well as hierarchical execution.
 * The only restriction is the parsec_handle_wait() before parsec_context_wait()
 */

double time_elapsed = 0.0;
double sync_time_elapsed = 0.0;

int count = 0;

enum regions {
               TILE_FULL,
             };

int
test_task( parsec_execution_unit_t    *context,
           parsec_execution_context_t *this_task )
{
    (void)context;

    int *amount_of_work;
    parsec_dtd_unpack_args( this_task,
                           UNPACK_VALUE,  &amount_of_work
                          );
    int i, j, bla;
    for( i = 0; i < *amount_of_work; i++ ) {
        //for( j = 0; j < *amount_of_work; j++ ) {
        for( j = 0; j < 2; j++ ) {
            bla = j*2;
            bla = j + 20;
            bla = j*2+i+j+i*i;
        }
    }
    count++;
    (void)bla;
    return PARSEC_HOOK_RETURN_DONE;
}

int
test_task_generator( parsec_execution_unit_t    *context,
                     parsec_execution_context_t *this_task )
{
    (void)context;

    tiled_matrix_desc_t *ddescB;
    int amount = 0, *nb, *nt;
    int rank = context->virtual_process->parsec_context->my_rank;
    int world = context->virtual_process->parsec_context->nb_nodes, i;

    parsec_dtd_unpack_args( this_task,
                            UNPACK_VALUE, &nb,
                            UNPACK_VALUE, &nt,
                            0
                          );

    ddescB = create_and_distribute_empty_data(rank, world, *nb, *nt);
    parsec_ddesc_set_key((parsec_ddesc_t *)ddescB, "B");
    parsec_ddesc_t *B = (parsec_ddesc_t *)ddescB;
    parsec_dtd_ddesc_init(B);

    parsec_handle_t *parsec_dtd_handle = parsec_dtd_handle_new();
    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( context->virtual_process->parsec_context, parsec_dtd_handle );

    for( i = 0; i < 100; i++ ) {
        parsec_insert_task( parsec_dtd_handle, test_task,    0,  "Test_Task",
                            sizeof(int),       &amount,    VALUE,
                            PASSED_BY_REF,     TILE_OF_KEY(B, rank),      INOUT | AFFINITY,
                            0 );
    }

    parsec_dtd_data_flush(parsec_dtd_handle, TILE_OF_KEY(B, rank));

    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_handle_wait( context->virtual_process->parsec_context, parsec_dtd_handle );

    parsec_dtd_ddesc_fini(B);
    free_data(ddescB);

    parsec_handle_free( parsec_dtd_handle );

    count++;

    return PARSEC_HOOK_RETURN_DONE;
}

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int rank, world, cores = 8;

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

    int m;
    int nb, nt;
    tiled_matrix_desc_t *ddescA;
    parsec_handle_t *parsec_dtd_handle;

    parsec = parsec_init( cores, &argc, &argv );

    parsec_dtd_handle = parsec_dtd_handle_new();

    /* Registering the dtd_handle with PARSEC context */
    parsec_enqueue( parsec, parsec_dtd_handle );
    parsec_context_start( parsec );

    nb = 1; /* size of each tile */
    nt = world; /* total tiles */

    ddescA = create_and_distribute_empty_data(rank, world, nb, nt);
    parsec_ddesc_set_key((parsec_ddesc_t *)ddescA, "A");

#if defined(PARSEC_HAVE_MPI)
    parsec_arena_construct(parsec_dtd_arenas[TILE_FULL],
                          nb*sizeof(int), PARSEC_ARENA_ALIGNMENT_SSE,
                          MPI_INT);
#endif

    parsec_ddesc_t *A = (parsec_ddesc_t *)ddescA;
    parsec_dtd_ddesc_init(A);

    SYNC_TIME_START();

    for( m = 0; m < nt; m++ ) {
        parsec_insert_task( parsec_dtd_handle, test_task_generator,    0,  "Test_Task_generator",
                            sizeof(int),       &nb,                 VALUE,
                            sizeof(int),       &nt,                 VALUE,
                            PASSED_BY_REF,     TILE_OF_KEY(A, m),   INOUT | AFFINITY,
                            0 );

        parsec_dtd_data_flush(parsec_dtd_handle, TILE_OF_KEY(A, m));
    }

    /* finishing all the tasks inserted, but not finishing the handle */
    parsec_dtd_handle_wait( parsec, parsec_dtd_handle );

    parsec_output( 0, "Successfully executed %d tasks in rank %d\n", count, parsec->my_rank );

    SYNC_TIME_PRINT(rank, ("\n") );

    parsec_context_wait(parsec);

    parsec_arena_destruct(parsec_dtd_arenas[0]);
    parsec_dtd_ddesc_fini( A );
    free_data(ddescA);

    parsec_handle_free( parsec_dtd_handle );

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return 0;
}