#define BENCHMARK "OSU MPI%s Allreduce Latency Test"
/*
 * Copyright (C) 2002-2016 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
/* 
 * Modified for PaRSEC Evaluation by A. Bouteiller
 * Copyright (C) 2018       The University of Tennessee and the University 
 * of Tennessee Research Fundation. ALL RIGHTS RESERVED.
 */
#include "osu_coll.h"

int osu_main(int argc, char *argv[], volatile int* stop)
{
    int i, numprocs, rank, size, gstop;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    float *sendbuf, *recvbuf;
    int po_ret;
    size_t bufsize;
    MPI_Comm world;

    set_header(HEADER);
    set_benchmark_name("osu_allreduce");
    enable_accel_support();
    po_ret = process_options(argc, argv);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_Comm_dup(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(world, &rank);
    MPI_Comm_size(world, &numprocs);

    switch (po_ret) {
        case po_bad_usage:
            print_bad_usage_message(rank);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        case po_help_message:
            print_help_message(rank);
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_version_message:
            print_version_message(rank);
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_okay:
            break;
    }

    if(numprocs < 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (options.max_message_size > options.max_mem_limit) {
        options.max_message_size = options.max_mem_limit;
    }

    options.min_message_size /= sizeof(float);
    if (options.min_message_size < DEFAULT_MIN_MESSAGE_SIZE) {
        options.min_message_size = DEFAULT_MIN_MESSAGE_SIZE;
    }

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&sendbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(world, EXIT_FAILURE);
    }
    set_buffer(sendbuf, options.accel, 1, bufsize);

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&recvbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(world, EXIT_FAILURE);
    }
    set_buffer(recvbuf, options.accel, 0, bufsize);

    print_preamble(rank);

 // a crude barrier to wait for parsec start
 do {} while(0 == stop);
 do {
    for(size=options.min_message_size; size*sizeof(float) <= options.max_message_size; size *= 2) {

        if(size > LARGE_MESSAGE_SIZE) {
            options.skip = options.skip_large;
            options.iterations = options.iterations_large;
        }

        MPI_Barrier(world);

        timer=0.0;
        for(i=0; i < options.iterations + options.skip ; i++) {
            t_start = MPI_Wtime();
            MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, world );
            t_stop=MPI_Wtime();
            if(i>=options.skip){

            timer+=t_stop-t_start;
            }
            MPI_Barrier(world);
        }
        latency = (double)(timer * 1e6) / options.iterations;

        MPI_Allreduce(stop, &gstop, 1, MPI_INT, MPI_MAX, world);
#if 1 
        if(2 == gstop) {
        //    printf("rank %04d bailout (stop=%d, gstop=%d) from IMB min=%d max=%d\n", rank, stop, gstop, options.min_message_size, options.max_message_size); fflush(stdout);
            goto bailout;
        }
#endif

        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                world);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                world);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                world);
        avg_time = avg_time/numprocs;

        print_stats(rank, size * sizeof(float), avg_time, min_time, max_time);
        MPI_Barrier(world);
    }
} while(1);
bailout:
    MPI_Barrier(world);
    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);

    if (none != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_Comm_free(&world);
    return EXIT_SUCCESS;
}
