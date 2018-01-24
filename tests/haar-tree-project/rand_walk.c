#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/arena.h"

#include "tree_dist.h"
#include "random_walk.h"

#include <unistd.h>
#include <string.h>

extern char *optarg;
extern int optind;
extern int optopt;
extern int opterr;
extern int optreset;

int main(int argc, char *argv[])
{
    parsec_context_t* parsec;
    int rc;
    int rank, world;
    two_dim_block_cyclic_t fakeDesc;
    parsec_random_walk_taskpool_t *rwalk;
    parsec_arena_t arena;
    int pargc = 0, i, dashdash = -1;
    char **pargv;
    int ret, ch;
    int depth = -1;
    double prob_branch = .5;
    int alrm = 0;
    int verbose = 0;

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
    MPI_Barrier(MPI_COMM_WORLD);

    if( depth == -1 ) {
        depth = 10 * world;
    }
    
    rwalk = parsec_random_walk_new(world, (parsec_data_collection_t*)&fakeDesc, depth, prob_branch, verbose);
    rwalk->arenas[PARSEC_random_walk_DEFAULT_ARENA] = &arena;
    rc = parsec_enqueue(parsec, &rwalk->super);
    PARSEC_CHECK_ERROR(rc, "parsec_enqueue");
    rc = parsec_context_start(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_start");
    rc = parsec_context_wait(parsec);
    PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
        
    rwalk->arenas[PARSEC_random_walk_DEFAULT_ARENA] = NULL;
    parsec_taskpool_free(&rwalk->super);
    ret = 0;

    parsec_fini(&parsec);

#ifdef PARSEC_HAVE_MPI
    MPI_Finalize();
#endif

    return ret;
}
