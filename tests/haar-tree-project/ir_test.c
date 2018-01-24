#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/arena.h"
#include "parsec/sys/atomic.h"

#include "tree_dist.h"
#include "random_walk.h"
#include "project.h"

#include <unistd.h>
#include <string.h>
#include <pthread.h>

extern char *optarg;
extern int optind;
extern int optopt;
extern int opterr;
extern int optreset;

static MPI_Comm ir_comm;

static int stop = 0;
static int send_buf = 0;
static int recv_buf;
static uint64_t ir_nb_sent = 0;
static double ir_delta = -1.0;

static void *ir_send(void *_)
{
    int i, r, w;
    double t1, t2;

    (void)_;

    MPI_Comm_size(ir_comm, &w);
    MPI_Comm_rank(ir_comm, &r);

    i = (r + 1) % w;
    t1 = MPI_Wtime();
    while(!stop) {
        do {
            i = (i+1) % w;
        } while(i == r);
        MPI_Send(&send_buf, 1, MPI_INT, i, 0, ir_comm);
        ir_nb_sent++;
    }
    t2 = MPI_Wtime();
    
    send_buf = 1;
    for(i = 0; i < w; i++) {
        if(i != r) {
            MPI_Send(&send_buf, 1, MPI_INT, i, 0, ir_comm);
        }
    }
    
    ir_delta = t2 - t1;
        
    return NULL;
}

static void *ir_recv(void *_)
{
    int w, r, n;

    (void)_;
    
    MPI_Comm_size(ir_comm, &w);
    MPI_Comm_rank(ir_comm, &r);

    n = 1;
    while(n != w) {
        MPI_Recv(&recv_buf, 1, MPI_INT, MPI_ANY_SOURCE, 0, ir_comm, MPI_STATUS_IGNORE);
        if( recv_buf == 1 ) {
            n++;
        } 
    }
    
    return NULL;
}

int osu_main(int argc, char *argv[], volatile int stop);
static void *ir_osu_allreduce(void*_) {
    int argc = 0;
    char * argv[9] = {{0}};
    argv[0] = strdup("osu_allreduce_mt");
    argv[1] = strdup("-f");
    argv[2] = strdup("-i");
    argv[3] = strdup("1000");
    argv[4] = strdup("-x");
    argv[5] = strdup("50");
    argv[6] = strdup("-m");
    argv[7] = strdup("1:1");
    osu_main(argc, argv, &stop);
    return NULL;
}

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rc;
    int rank, world;
    two_dim_block_cyclic_t fakeDesc;
    parsec_random_walk_taskpool_t *rwalk;
    parsec_project_taskpool_t *proj1, *proj2;
    parsec_arena_t arena;
    int pargc = 0, i, dashdash = -1;
    char **pargv;
    int ret, ch;
    int depth = -1;
    double prob_branch = .99;
    int alrm = 0;
    int verbose = 0;
    pthread_t th0, th1;
    tree_dist_t *treeA;

#if defined(PARSEC_HAVE_MPI)
    {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        printf("provided = %d, MPI_THREAD_MULTIPLE = %d\n", provided, MPI_THREAD_MULTIPLE);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    world = 1;
    rank = 0;
#endif

    for(i = 1; i < argc; i++) {
        if( strcmp(argv[i], "--") == 0 ) {
            dashdash = i;
            pargc = 0;
        } else if( dashdash != -1 ) {
            pargc++;
        }
    }
    pargv = malloc( (pargc+1) * sizeof(char*));
    if( dashdash != -1 ) {
        for(i = dashdash+1; i < argc; i++) {
            pargv[i-dashdash-1] = strdup(argv[i]);
        }
        pargv[i-dashdash-1] = NULL;
    } else {
        pargv[0] = NULL;
    }
    parsec = parsec_init(1, &pargc, &pargv);
    
    while ((ch = getopt(argc, argv, "l:p:A:v")) != -1) {
        switch (ch) {
        case 'A':
            alrm = atoi(optarg);
            break;
        case 'l':
            depth = atoi(optarg);
            break;
        case 'p':
            prob_branch = strtod(optarg, NULL);
            break;
        case 'v':
            verbose = 1;
            break;
        case '?':
        default:
            fprintf(stderr,
                    "Usage: %s [-l depth] [-p branch prob] [-A alarm] [-v] -- <parsec arguments>\n"
                    "   Implement the a random walk of length l with a branching probabililty of p\n"
                    "   (v is for verbose, A to set an alarm to prevent the bench to run too long)\n",
                    argv[0]);
            exit(1);
        }
    }

    two_dim_block_cyclic_init(&fakeDesc, matrix_RealFloat, matrix_Tile,
                              world, rank, 1, 1, world, world, 0, 0, world, world, 1, 1, 1);

    parsec_arena_construct( &arena,
                            parsec_datadist_getsizeoftype(matrix_RealFloat),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_float_t
                         );    
    
    if(alrm > 0) {
        alarm(alrm);
    }
#if USING_IRTH
    MPI_Comm_dup(MPI_COMM_WORLD, &ir_comm);

    pthread_create(&th0, NULL, ir_recv, NULL);

    MPI_Barrier(MPI_COMM_WORLD);
    
    pthread_create(&th1, NULL, ir_send, NULL);
#else
    pthread_create(&th0, NULL, ir_osu_allreduce, NULL);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if( depth == -1 ) {
        depth = 10 * world;
    }

    treeA = tree_dist_create_empty(rank, world);
    parsec_arena_construct( &arena,
                           2 * parsec_datadist_getsizeoftype(matrix_RealFloat),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_float_t
                         );

    MPI_Barrier(MPI_COMM_WORLD);
 
    rwalk = parsec_random_walk_new(world, (parsec_data_collection_t*)&fakeDesc, depth, prob_branch, verbose);
    rwalk->arenas[PARSEC_random_walk_DEFAULT_ARENA] = &arena;
    rc = parsec_enqueue(parsec, &rwalk->super);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");
#if 0
    proj1 = parsec_project_new(treeA, world, (parsec_data_collection_t*)&fakeDesc, 1e-13, verbose, 0.33333);
    proj1->arenas[PARSEC_project_DEFAULT_ARENA] = &arena;
    rc = parsec_enqueue(parsec, &proj1->super);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");
    proj2 = parsec_project_new(treeA, world, (parsec_data_collection_t*)&fakeDesc, 1e-13, verbose, 0.66666);
    rc = parsec_enqueue(parsec, &proj2->super);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");
#endif
    parsec_atomic_fetch_inc_int32(&stop);
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");

    parsec_atomic_fetch_inc_int32(&stop);
    pthread_join(th0, NULL);
#if USE_IRTH
    pthread_join(th1, NULL);

    printf("Injection rate at rank %d: %lu messages sent in %g seconds : %g msg/s\n", rank, ir_nb_sent, ir_delta, ir_nb_sent / ir_delta);
#endif
    
    rwalk->arenas[PARSEC_random_walk_DEFAULT_ARENA] = NULL;
    parsec_taskpool_free(&rwalk->super);
    ret = 0;

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return ret;
}
