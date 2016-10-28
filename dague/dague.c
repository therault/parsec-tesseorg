/**
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague_internal.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <limits.h>
#if defined(DAGUE_HAVE_GEN_H)
#include <libgen.h>
#endif  /* defined(DAGUE_HAVE_GEN_H) */
#if defined(DAGUE_HAVE_GETOPT_H)
#include <getopt.h>
#endif  /* defined(DAGUE_HAVE_GETOPT_H) */
#include <dague/ayudame.h>

#include "dague/mca/pins/pins.h"
#include "dague/mca/sched/sched.h"
#include "dague/utils/output.h"
#include "dague/data_internal.h"
#include "dague/class/list.h"
#include "dague/scheduling.h"
#include "dague/class/barrier.h"
#include "dague/remote_dep.h"
#include "dague/datarepo.h"
#include "dague/bindthread.h"
#include "dague/dague_prof_grapher.h"
#include "dague/vpmap.h"
#include "dague/utils/mca_param.h"
#include "dague/utils/installdirs.h"
#include "dague/devices/device.h"
#include "dague/utils/cmd_line.h"
#include "dague/utils/mca_param_cmd_line.h"

#include "dague/mca/mca_repository.h"

#ifdef DAGUE_PROF_TRACE
#include "dague/profiling.h"
#endif

#include "dague/dague_hwloc.h"
#ifdef DAGUE_HAVE_HWLOC
#include "dague/hbbuffer.h"
#endif

#ifdef DAGUE_HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

/**
 * Global variables.
 */
size_t dague_task_startup_iter = 64;
size_t dague_task_startup_chunk = 256;

dague_data_allocate_t dague_data_allocate = malloc;
dague_data_free_t     dague_data_free = free;

#if defined(DAGUE_PROF_TRACE)
#if defined(DAGUE_PROF_TRACE_SCHEDULING_EVENTS)
int MEMALLOC_start_key, MEMALLOC_end_key;
int schedule_poll_begin, schedule_poll_end;
int schedule_push_begin, schedule_push_end;
int schedule_sleep_begin, schedule_sleep_end;
int queue_add_begin, queue_add_end;
int queue_remove_begin, queue_remove_end;
#endif  /* defined(DAGUE_PROF_TRACE_SCHEDULING_EVENTS) */
int device_delegate_begin, device_delegate_end;
int arena_memory_alloc_key, arena_memory_free_key;
int arena_memory_used_key, arena_memory_unused_key;
int task_memory_alloc_key, task_memory_free_key;
#endif  /* DAGUE_PROF_TRACE */

#ifdef DAGUE_HAVE_HWLOC
#define MAX_CORE_LIST 128
#endif

int dague_want_rusage = 0;
#if defined(DAGUE_HAVE_GETRUSAGE) && !defined(__bgp__)
#include <sys/time.h>
#include <sys/resource.h>

static struct rusage _dague_rusage;

static char *dague_enable_dot = NULL;
static char *dague_app_name = NULL;
static dague_device_t* dague_device_cpus = NULL;
static dague_device_t* dague_device_recursive = NULL;

static int dague_runtime_max_number_of_cores = -1;
static int dague_runtime_bind_main_thread = 1;

/**
 * Object based task definition (no specialized constructor and destructor) */
OBJ_CLASS_INSTANCE(dague_execution_context_t, dague_hashtable_item_t,
                   NULL, NULL);

static void dague_rusage(bool print)
{
    struct rusage current;
    getrusage(RUSAGE_SELF, &current);
    if( print ) {
        double usr, sys;

        usr = ((current.ru_utime.tv_sec - _dague_rusage.ru_utime.tv_sec) +
               (current.ru_utime.tv_usec - _dague_rusage.ru_utime.tv_usec) / 1000000.0);
        sys = ((current.ru_stime.tv_sec - _dague_rusage.ru_stime.tv_sec) +
               (current.ru_stime.tv_usec - _dague_rusage.ru_stime.tv_usec) / 1000000.0);

        dague_inform("==== Resource Usage Data...\n"
                     "-------------------------------------------------------------\n"
                     "User Time   (secs)          : %10.3f\n"
                     "System Time (secs)          : %10.3f\n"
                     "Total Time  (secs)          : %10.3f\n"
                     "Minor Page Faults           : %10ld\n"
                     "Major Page Faults           : %10ld\n"
                     "Swap Count                  : %10ld\n"
                     "Voluntary Context Switches  : %10ld\n"
                     "Involuntary Context Switches: %10ld\n"
                     "Block Input Operations      : %10ld\n"
                     "Block Output Operations     : %10ld\n"
                     "=============================================================\n",
                     usr, sys, usr + sys,
                     current.ru_minflt  - _dague_rusage.ru_minflt, current.ru_majflt  - _dague_rusage.ru_majflt,
                     current.ru_nswap   - _dague_rusage.ru_nswap,
                     current.ru_nvcsw   - _dague_rusage.ru_nvcsw, current.ru_nivcsw  - _dague_rusage.ru_nivcsw,
                     current.ru_inblock - _dague_rusage.ru_inblock, current.ru_oublock - _dague_rusage.ru_oublock);
    }
    _dague_rusage = current;
    return;
}
#define dague_rusage(b) do { if(dague_want_rusage > 0) dague_rusage(b); } while(0)
#else
#define dague_rusage(b) do {} while(0)
#endif /* defined(DAGUE_HAVE_GETRUSAGE) */

static void dague_handle_empty_repository(void);

typedef struct __dague_temporary_thread_initialization_t {
    dague_vp_t *virtual_process;
    int th_id;
    int nb_cores;
    int bindto;
    int bindto_ht;
    dague_barrier_t*  barrier;       /**< the barrier used to synchronize for the
                                      *   local VP data construction. */
} __dague_temporary_thread_initialization_t;

static int dague_parse_binding_parameter(void * optarg, dague_context_t* context,
                                         __dague_temporary_thread_initialization_t* startup);
static int dague_parse_comm_binding_parameter(void * optarg, dague_context_t* context);

static void* __dague_thread_init( __dague_temporary_thread_initialization_t* startup )
{
    dague_execution_unit_t* eu;
    int pi;

    /* don't use DAGUE_THREAD_IS_MASTER, it is too early and we cannot yet allocate the eu struct */
    if( (0 != startup->virtual_process) || (0 != startup->th_id) || dague_runtime_bind_main_thread ) {
        /* Bind to the specified CORE */
        dague_bindthread(startup->bindto, startup->bindto_ht);
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Bind thread %i.%i on core %i [HT %i]",
                            startup->virtual_process->vp_id, startup->th_id,
                            startup->bindto, startup->bindto_ht);
    } else {
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Don't bind the main thread %i.%i",
                            startup->virtual_process->vp_id, startup->th_id);
    }

    eu = (dague_execution_unit_t*)malloc(sizeof(dague_execution_unit_t));
    if( NULL == eu ) {
        return NULL;
    }
    eu->th_id            = startup->th_id;
    eu->virtual_process  = startup->virtual_process;
    eu->scheduler_object = NULL;
    startup->virtual_process->execution_units[startup->th_id] = eu;
    eu->core_id          = startup->bindto;
#if defined(DAGUE_HAVE_HWLOC)
    eu->socket_id        = dague_hwloc_socket_id(startup->bindto);
#else
    eu->socket_id        = 0;
#endif  /* defined(DAGUE_HAVE_HWLOC) */

#if defined(DAGUE_SCHED_REPORT_STATISTICS)
    eu->sched_nb_tasks_done = 0;
#endif

    /**
     * A single thread per VP has a little bit more responsability: allocating
     * the memory pools.
     */
    if( startup->th_id == (startup->nb_cores - 1) ) {
        dague_vp_t *vp = startup->virtual_process;
        dague_execution_context_t fake_context;
        data_repo_entry_t fake_entry;
        dague_mempool_construct( &vp->context_mempool,
                                 OBJ_CLASS(dague_execution_context_t), sizeof(dague_execution_context_t),
                                 ((char*)&fake_context.super.mempool_owner) - ((char*)&fake_context),
                                 vp->nb_cores );

        for(pi = 0; pi <= MAX_PARAM_COUNT; pi++)
            dague_mempool_construct( &vp->datarepo_mempools[pi],
                                     NULL, sizeof(data_repo_entry_t)+(pi-1)*sizeof(dague_arena_chunk_t*),
                                     ((char*)&fake_entry.data_repo_mempool_owner) - ((char*)&fake_entry),
                                     vp->nb_cores);

    }
    /* Synchronize with the other threads */
    dague_barrier_wait(startup->barrier);

    if( NULL != current_scheduler->module.flow_init )
        current_scheduler->module.flow_init(eu, startup->barrier);

    eu->context_mempool = &(eu->virtual_process->context_mempool.thread_mempools[eu->th_id]);
    for(pi = 0; pi <= MAX_PARAM_COUNT; pi++)
        eu->datarepo_mempools[pi] = &(eu->virtual_process->datarepo_mempools[pi].thread_mempools[eu->th_id]);

#ifdef DAGUE_PROF_TRACE
    {
        char *binding = dague_hwloc_get_binding();
        eu->eu_profile = dague_profiling_thread_init( 2*1024*1024,
                                                      DAGUE_PROFILE_THREAD_STR,
                                                      eu->th_id,
                                                      eu->virtual_process->vp_id,
                                                      binding);
        free(binding);
    }
    if( NULL != eu->eu_profile ) {
        PROFILING_THREAD_SAVE_iINFO(eu->eu_profile, "boundto", startup->bindto);
        PROFILING_THREAD_SAVE_iINFO(eu->eu_profile, "th_id", eu->th_id);
        PROFILING_THREAD_SAVE_iINFO(eu->eu_profile, "vp_id", eu->virtual_process->vp_id );
    }
#endif /* DAGUE_PROF_TRACE */

    PINS_THREAD_INIT(eu);

#if defined(DAGUE_SIM)
    eu->largest_simulation_date = 0;
#endif

    /* The main thread of VP 0 will go back to the user level */
    if( DAGUE_THREAD_IS_MASTER(eu) ) {
        return NULL;
    }

    return (void*)(long)__dague_context_wait(eu);
}

static void dague_vp_init( dague_vp_t *vp,
                           int32_t nb_cores,
                           __dague_temporary_thread_initialization_t *startup)
{
    int t, pi;
    dague_barrier_t*  barrier;

    vp->nb_cores = nb_cores;

    barrier = (dague_barrier_t*)malloc(sizeof(dague_barrier_t));
    dague_barrier_init(barrier, NULL, vp->nb_cores);

    /* Prepare the temporary storage for each thread startup */
    for( t = 0; t < vp->nb_cores; t++ ) {
        startup[t].th_id = t;
        startup[t].virtual_process = vp;
        startup[t].nb_cores = nb_cores;
        startup[t].bindto = -1;
        startup[t].bindto_ht = -1;
        startup[t].barrier = barrier;
        pi = vpmap_get_nb_cores_affinity(vp->vp_id, t);
        if( 1 == pi )
            vpmap_get_core_affinity(vp->vp_id, t, &startup[t].bindto, &startup[t].bindto_ht);
        else if( 1 < pi )
            dague_warning("multiple core to bind on... for now, do nothing"); //TODO: what does that mean?
    }
}

#define DEFAULT_APPNAME "app_name_%d"

#define GET_INT_ARGV(CMD, ARGV, VALUE) \
do { \
    int __nb_elems = dague_cmd_line_get_ninsts((CMD), (ARGV)); \
    if( 0 != __nb_elems ) { \
        char* __value = dague_cmd_line_get_param((CMD), (ARGV), 0, 0); \
        if( NULL != __value ) \
            (VALUE) = (int)strtol(__value, NULL, 10); \
    } \
} while (0)

#define GET_STR_ARGV(CMD, ARGV, VALUE) \
do { \
    int __nb_elems = dague_cmd_line_get_ninsts((CMD), (ARGV)); \
    if( 0 != __nb_elems ) { \
        (VALUE) = dague_cmd_line_get_param((CMD), (ARGV), 0, 0); \
    } \
} while (0)


dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[] )
{
    int ret, nb_vp, p, t, nb_total_comp_threads, display_vpmap = 0;
    char *comm_binding_parameter = NULL;
    char *binding_parameter = NULL;
    __dague_temporary_thread_initialization_t *startup;
    dague_context_t* context;
    dague_cmd_line_t *cmd_line = NULL;
    char **environ = NULL;
    char **env_variable, *env_name, *env_value;
    char *dague_enable_profiling = NULL;  /* profiling file prefix when DAGUE_PROF_TRACE is on */
    int slow_option_warning = 0;

    dague_installdirs_open();
    dague_mca_param_init();
    dague_output_init();

    /* Extract what we can from the arguments */
    cmd_line = OBJ_NEW(dague_cmd_line_t);
    if( NULL == cmd_line ) {
        return NULL;
    }

    /* Declare the command line for the .dot generation */
    dague_cmd_line_make_opt3(cmd_line, 'h', "help", "help", 0,
                             "Show the usage text.");
    dague_cmd_line_make_opt3(cmd_line, '.', "dot", "dague_dot", 1,
                             "Filename for the .dot file");
    dague_cmd_line_make_opt3(cmd_line, 'b', NULL, "dague_bind", 1,
                             "Execution thread binding");
    dague_cmd_line_make_opt3(cmd_line, 'C', NULL, "dague_bind_comm", 1,
                             "Communication thread binding");
    dague_cmd_line_make_opt3(cmd_line, 'c', "cores", "cores", 1,
                             "Number of cores to used");
    dague_cmd_line_make_opt3(cmd_line, 'g', "gpus", "gpus", 1,
                             "Number of GPU to used (deprecated use MCA instead)");
    dague_cmd_line_make_opt3(cmd_line, 'V', "vpmap", "vpmap", 1,
                             "Virtual process map");
    dague_cmd_line_make_opt3(cmd_line, 'H', "ht", "ht", 1,
                             "Enable hyperthreading");
    dague_mca_cmd_line_setup(cmd_line);

    if( (NULL != pargc) && (0 != *pargc) ) {
        dague_app_name = strdup( (*pargv)[0] );

        ret = dague_cmd_line_parse(cmd_line, true, *pargc, *pargv);
        if (DAGUE_SUCCESS != ret) {
            fprintf(stderr, "%s: command line error (%d)\n", (*pargv)[0], ret);
        }
    } else {
        ret = asprintf( &dague_app_name, DEFAULT_APPNAME, (int)getpid() );
        if (ret == -1) {
            dague_app_name = strdup( "app_name_XXXXXX" );
        }
    }

    ret = dague_mca_cmd_line_process_args(cmd_line, &environ, &environ);
    if( environ != NULL ) {
        for(env_variable = environ;
            *env_variable != NULL;
            env_variable++) {
            env_name = *env_variable;
            for(env_value = env_name; *env_value != '\0' && *env_value != '='; env_value++)
                /* nothing */;
            if(*env_value == '=') {
                *env_value = '\0';
                env_value++;
            }
            setenv(env_name, env_value, 1);
            free(*env_variable);
        }
        free(environ);
    }
    dague_debug_init();
    mca_components_repository_init();

#if defined(DAGUE_HAVE_HWLOC)
    dague_hwloc_init();
#endif  /* defined(HWLOC) */

    if( dague_cmd_line_is_taken(cmd_line, "ht") ) {
#if defined(DAGUE_HAVE_HWLOC)
        int hyperth = 0;
        GET_INT_ARGV(cmd_line, "ht", hyperth);
        dague_hwloc_allow_ht(hyperth);
#else
        dague_warning("Option ht (hyper-threading) is only supported when HWLOC is enabled at compile time.");
#endif  /* defined(DAGUE_HAVE_HWLOC) */
    }

    /* Set a default the number of cores if not defined by parameters
     * - with hwloc if available
     * - with sysconf otherwise (hyperthreaded core number)
     */
    dague_mca_param_reg_int_name("runtime", "num_cores", "The total number of cores to be used by the runtime (-1 for all available)",
                                 false, false, dague_runtime_max_number_of_cores, &dague_runtime_max_number_of_cores);
    if( nb_cores <= 0 ) {
        if( -1 == dague_runtime_max_number_of_cores )
            nb_cores = dague_hwloc_nb_real_cores();
        else {
            nb_cores = dague_runtime_max_number_of_cores;
            if( dague_runtime_max_number_of_cores > dague_hwloc_nb_real_cores() ) {
                dague_warning("The runtime is running in an over-subscribe mode, usually unfit for performance.");
            }
        }
    }
    dague_mca_param_reg_int_name("runtime", "bind_main_thread", "Force the binding of the thread calling dague_init",
                                 false, false, dague_runtime_bind_main_thread, &dague_runtime_bind_main_thread);

    if( dague_cmd_line_is_taken(cmd_line, "gpus") ) {
        dague_warning("Option g (for accelerators) is deprecated as an argument. Use the MCA parameter instead.");
    }

    /* Allow the dague_init arguments to overwrite all the previous behaviors */
    GET_INT_ARGV(cmd_line, "cores", nb_cores);
    GET_STR_ARGV(cmd_line, "dague_bind_comm", comm_binding_parameter);
    GET_STR_ARGV(cmd_line, "dague_bind", binding_parameter);

    /**
     * Initialize the VPMAP, the discrete domains hosting
     * execution flows but where work stealing is prevented.
     */
    {
        /* Change the vpmap choice: first cancel the previous one if any */
        vpmap_fini();

        if( dague_cmd_line_is_taken(cmd_line, "vpmap") ) {
            char* optarg = NULL;
            GET_STR_ARGV(cmd_line, "vpmap", optarg);
            if( !strncmp(optarg, "display", 7 )) {
                display_vpmap = 1;
            } else {
                if( !strncmp(optarg, "flat", 4) ) {
                    /* default case (handled in dague_init) */
                } else if( !strncmp(optarg, "hwloc", 5) ) {
                    vpmap_init_from_hardware_affinity(nb_cores);
                } else if( !strncmp(optarg, "file:", 5) ) {
                    vpmap_init_from_file(optarg + 5);
                } else if( !strncmp(optarg, "rr:", 3) ) {
                    int n, p, co;
                    sscanf(optarg, "rr:%d:%d:%d", &n, &p, &co);
                    vpmap_init_from_parameters(n, p, co);
                } else {
                    dague_warning("VPMAP choice (-V argument): %s is invalid. Falling back to default!", optarg);
                }
            }
        }
        nb_vp = vpmap_get_nb_vp();
        if( -1 == nb_vp ) {
            vpmap_init_from_flat(nb_cores);
            nb_vp = vpmap_get_nb_vp();
        }
    }

    if( dague_cmd_line_is_taken(cmd_line, "dot") ) {
        char* optarg = NULL;
        GET_STR_ARGV(cmd_line, "dot", optarg);

        if( dague_enable_dot ) free( dague_enable_dot );
        if( NULL == optarg ) {
            dague_enable_dot = strdup(dague_app_name);
        } else {
            dague_enable_dot = strdup(optarg);
        }
    }

    /* the extra allocation will pertain to the virtual_processes array */
    context = (dague_context_t*)malloc(sizeof(dague_context_t) + (nb_vp-1) * sizeof(dague_vp_t*));

    context->__dague_internal_finalization_in_progress = 0;
    context->__dague_internal_finalization_counter = 0;
    context->active_objects      = 0;
    context->flags               = 0;
    context->nb_nodes            = 1;
    context->comm_ctx            = NULL;
    context->my_rank             = 0;
    context->nb_vp               = nb_vp;
#if defined(DAGUE_SIM)
    context->largest_simulation_date = 0;
#endif /* DAGUE_SIM */
#if defined(DAGUE_HAVE_HWLOC)
    context->cpuset_allowed_mask = NULL;
    context->cpuset_free_mask    = NULL;
    context->comm_th_core        = -1;
#endif  /* defined(DAGUE_HAVE_HWLOC) */

    /* TODO: nb_cores should depend on the vp_id */
    nb_total_comp_threads = 0;
    for(p = 0; p < nb_vp; p++) {
        nb_total_comp_threads += vpmap_get_nb_threads_in_vp(p);
    }

    if( nb_cores != nb_total_comp_threads ) {
        dague_warning("Your vpmap uses %d threads when %d cores where available",
                     nb_total_comp_threads, nb_cores);
        nb_cores = nb_total_comp_threads;
    }

    startup = (__dague_temporary_thread_initialization_t*)
        malloc(nb_total_comp_threads * sizeof(__dague_temporary_thread_initialization_t));

    t = 0;
    for( p = 0; p < nb_vp; p++ ) {
        dague_vp_t *vp;
        vp = (dague_vp_t *)malloc(sizeof(dague_vp_t) + (vpmap_get_nb_threads_in_vp(p)-1) * sizeof(dague_execution_unit_t*));
        vp->dague_context = context;
        vp->vp_id = p;
        context->virtual_processes[p] = vp;
        /**
         * Set the threads local variables from startup[t] -> startup[t+nb_cores].
         * Do not create or initialize any memory yet, or it will be automatically
         * bound to the allocation context of this thread.
         */
        dague_vp_init(vp, vpmap_get_nb_threads_in_vp(p), &(startup[t]));
        t += vp->nb_cores;
    }

    /**
     * Parameters defining the default ARENA behavior. Handle with care they can lead to
     * significant memory consumption or to a significant overhead in memory management
     * (allocation/deallocation). These values are only used by ARENAs constructed with
     * the default constructor (not the extended one).
     */
    dague_mca_param_reg_sizet_name("arena", "max_used", "The maximum amount of memory each arena can"
                                   " allocate (default unlimited)",
                                   false, false, dague_arena_max_allocated_memory, &dague_arena_max_allocated_memory);
    dague_mca_param_reg_sizet_name("arena", "max_cached", "The maxmimum amount of memory each arena can"
                                   " cache in a freelist (0=no caching)",
                                   false, false, dague_arena_max_cached_memory, &dague_arena_max_cached_memory);

    dague_mca_param_reg_sizet_name("task", "startup_iter", "The number of ready tasks to be generated during the startup "
                                   "before allowing the scheduler to distribute them across the entire execution context.",
                                   false, false, dague_task_startup_iter, &dague_task_startup_iter);
    dague_mca_param_reg_sizet_name("task", "startup_chunk", "The total number of tasks to be generated during the startup "
                                   "before delaying the remaining of the startup. The startup process will be "
                                   "continued at a later moment once the number of ready tasks decreases.",
                                   false, false, dague_task_startup_chunk, &dague_task_startup_chunk);

    dague_mca_param_reg_string_name("profile", "filename",
#if defined(DAGUE_PROF_TRACE)
                                    "Path to the profiling file (<none> to disable, <app> for app name, <*> otherwise)",
                                    false, false,
#else
                                    "Path to the profiling file (unused due to profiling being turned off during building)",
                                    false, true,  /* profiling disabled: read-only */
#endif  /* defined(DAGUE_PROF_TRACE) */
                                    "<none>", &dague_enable_profiling);
#if defined(DAGUE_PROF_TRACE)
    if( (0 != strncasecmp(dague_enable_profiling, "<none>", 6)) && (0 == dague_profiling_init( )) ) {
        int i, l;
        char *cmdline_info = basename(dague_app_name);

        /* Use either the app name (argv[0]) or the user provided filename */
        if( 0 == strncmp(dague_enable_profiling, "<app>", 5) ) {
            ret = dague_profiling_dbp_start( cmdline_info, dague_app_name );
        } else {
            ret = dague_profiling_dbp_start( dague_enable_profiling, dague_app_name );
        }
        if( ret != 0 ) {
            dague_warning("Profiling framework deactivated because of error %s.", dague_profiling_strerror());
        }

        l = 0;
        for(i = 0; i < *pargc; i++) {
            l += strlen( (*pargv)[i] ) + 1;
        }
        cmdline_info = (char*)calloc(sizeof(char), l + 1);
        l = 0;
        for(i = 0; i < *pargc; i++) {
            sprintf(cmdline_info + l, "%s ", (*pargv)[i]);
            l += strlen( (*pargv)[i] ) + 1;
        }
        cmdline_info[l] = '\0';
        dague_profiling_add_information("CMDLINE", cmdline_info);

        /* we should be adding the PaRSEC options to the profile here
         * instead of in common.c/h as we do now. */
        PROFILING_SAVE_iINFO("nb_cores", nb_cores);
        PROFILING_SAVE_iINFO("nb_vps", nb_vp);

        free(cmdline_info);

#  if defined(DAGUE_PROF_TRACE_SCHEDULING_EVENTS)
        dague_profiling_add_dictionary_keyword( "MEMALLOC", "fill:#FF00FF",
                                                0, NULL,
                                                &MEMALLOC_start_key, &MEMALLOC_end_key);
        dague_profiling_add_dictionary_keyword( "Sched POLL", "fill:#8A0886",
                                                0, NULL,
                                                &schedule_poll_begin, &schedule_poll_end);
        dague_profiling_add_dictionary_keyword( "Sched PUSH", "fill:#F781F3",
                                                0, NULL,
                                                &schedule_push_begin, &schedule_push_end);
        dague_profiling_add_dictionary_keyword( "Sched SLEEP", "fill:#FA58F4",
                                                0, NULL,
                                                &schedule_sleep_begin, &schedule_sleep_end);
        dague_profiling_add_dictionary_keyword( "Queue ADD", "fill:#767676",
                                                0, NULL,
                                                &queue_add_begin, &queue_add_end);
        dague_profiling_add_dictionary_keyword( "Queue REMOVE", "fill:#B9B243",
                                                0, NULL,
                                                &queue_remove_begin, &queue_remove_end);
#  endif /* DAGUE_PROF_TRACE_SCHEDULING_EVENTS */
#if defined(DAGUE_PROF_TRACE_ACTIVE_ARENA_SET)
        dague_profiling_add_dictionary_keyword( "ARENA_MEMORY", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &arena_memory_alloc_key, &arena_memory_free_key);
        dague_profiling_add_dictionary_keyword( "ARENA_ACTIVE_SET", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &arena_memory_used_key, &arena_memory_unused_key);
#endif  /* defined(DAGUE_PROF_TRACE_ACTIVE_ARENA_SET) */
        dague_profiling_add_dictionary_keyword( "TASK_MEMORY", "fill:#B9B243",
                                                sizeof(size_t), "size{int64_t}",
                                                &task_memory_alloc_key, &task_memory_free_key);
        dague_profiling_add_dictionary_keyword( "Device delegate", "fill:#EAE7C6",
                                                0, NULL,
                                                &device_delegate_begin, &device_delegate_end);
        /* Ready to rock! The profiling must be on by default */
        dague_profiling_start();
    }
#endif  /* DAGUE_PROF_TRACE */
    if (dague_enable_profiling) {
        free(dague_enable_profiling);
    }

    /* Extract the expected thread placement */
    if( NULL != comm_binding_parameter )
        dague_parse_comm_binding_parameter(comm_binding_parameter, context);
    dague_parse_binding_parameter(binding_parameter, context, startup);

    /* Introduce communication engine */
    (void)dague_remote_dep_init(context);

    /* Initialize Performance Instrumentation (PINS) */
    PINS_INIT(context);

    if(dague_enable_dot) {
#if defined(DAGUE_PROF_GRAPHER)
        dague_prof_grapher_init(dague_enable_dot, nb_total_comp_threads);
        slow_option_warning = 1;
#else
        dague_warning("DOT generation requested, but DAGUE_PROF_GRAPHER was not selected during compilation: DOT generation ignored.");
#endif  /* defined(DAGUE_PROF_GRAPHER) */
    }

#if defined(DAGUE_DEBUG_NOISIER) || defined(DAGUE_DEBUG_PARANOID)
    slow_option_warning = 1;
#endif
    if( slow_option_warning && 0 == context->my_rank ) {
        dague_warning("/!\\ THE PERFORMANCE OF THIS RUN ARE NOT OPTIMAL /!\\.\n");
        dague_debug_verbose(4, dague_debug_output, "--- compiled with DEBUG_NOISIER, DEBUG_PARANOID, or DOT generation requested.");
    }

    dague_devices_init(context);
    /* By now let's add one device for the CPUs */
    {
        dague_device_cpus = (dague_device_t*)calloc(1, sizeof(dague_device_t));
        dague_device_cpus->name = "default";
        dague_device_cpus->type = DAGUE_DEV_CPU;
        dague_devices_add(context, dague_device_cpus);
        /* TODO: This is plain WRONG, but should work by now */
        dague_device_cpus->device_sweight = nb_total_comp_threads * 8 * (float)2.27;
        dague_device_cpus->device_dweight = nb_total_comp_threads * 4 * 2.27;
    }
    /* By now let's add one device for the recursive kernels */
    {
        dague_device_recursive = (dague_device_t*)calloc(1, sizeof(dague_device_t));
        dague_device_recursive->name = "recursive";
        dague_device_recursive->type = DAGUE_DEV_RECURSIVE;
        dague_devices_add(context, dague_device_recursive);
        /* TODO: This is plain WRONG, but should work by now */
        dague_device_recursive->device_sweight = nb_total_comp_threads * 8 * (float)2.27;
        dague_device_recursive->device_dweight = nb_total_comp_threads * 4 * 2.27;
    }
    dague_devices_select(context);
    dague_devices_freeze(context);

    /* Init the data infrastructure. Must be done only after the freeze of the devices */
    dague_data_init(context);

    /* Initialize the barriers */
    dague_barrier_init( &(context->barrier), NULL, nb_total_comp_threads );

    /* Load the default scheduler. User can change it afterward,
     * but we need to ensure that one is loadable and here.
     */
    if( 0 == dague_set_scheduler( context ) ) {
        /* TODO: handle memory leak / thread leak here: this is a fatal
         * error for PaRSEC */
        dague_abort("Unable to load any scheduler in init function.");
        return NULL;
    }

    if( nb_total_comp_threads > 1 ) {
        pthread_attr_t thread_attr;

        pthread_attr_init(&thread_attr);
        pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
#ifdef __linux
        pthread_setconcurrency(nb_total_comp_threads);
#endif  /* __linux */

        context->pthreads = (pthread_t*)malloc(nb_total_comp_threads * sizeof(pthread_t));

        /* The first execution unit is for the master thread */
        for( t = 1; t < nb_total_comp_threads; t++ ) {
            pthread_create( &((context)->pthreads[t]),
                            &thread_attr,
                            (void* (*)(void*))__dague_thread_init,
                            (void*)&(startup[t]));
        }
    } else {
        context->pthreads = NULL;
    }

    __dague_thread_init( &startup[0] );

    /* Wait until all threads are done binding themselves */
    dague_barrier_wait( &(context->barrier) );
    context->__dague_internal_finalization_counter++;

    /* Release the temporary array used for starting up the threads */
    {
        dague_barrier_t* barrier = startup[0].barrier;
        dague_barrier_destroy(barrier);
        free(barrier);
        for(t = 0; t < nb_total_comp_threads; t++) {
            if(barrier != startup[t].barrier) {
                barrier = startup[t].barrier;
                dague_barrier_destroy(barrier);
                free(barrier);
            }
        }
    }
    free(startup);

    if( display_vpmap )
        vpmap_display_map();

    dague_mca_param_reg_int_name("profile", "rusage", "Report 'getrusage' satistics.\n"
            "0: no report, 1: per process report, 2: per thread report (if available).\n",
            false, false, dague_want_rusage, &dague_want_rusage);
    dague_rusage(false);

    AYU_INIT();

    if( dague_cmd_line_is_taken(cmd_line, "help") ||
        dague_cmd_line_is_taken(cmd_line, "h")) {
#if defined(DISTRIBUTED)
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if( 0 == rank )
#endif
        {
            char* help_msg = dague_cmd_line_get_usage_msg(cmd_line);
            dague_list_t* l = NULL;

            dague_output(0, "%s\n\nRegistered MCA parameters", help_msg);
            free(help_msg);

            dague_mca_param_dump(&l, 1);
            dague_mca_show_mca_params(l, "all", "all", 1);
            dague_mca_param_dump_release(l);
        }
        dague_fini(&context);
        return NULL;
    }

    if( NULL != cmd_line )
        OBJ_RELEASE(cmd_line);

    return context;
}

static void dague_vp_fini( dague_vp_t *vp )
{
    int i;
    dague_mempool_destruct( &vp->context_mempool );
    for(i = 0; i <= MAX_PARAM_COUNT; i++)
        dague_mempool_destruct( &vp->datarepo_mempools[i]);

    for(i = 0; i < vp->nb_cores; i++) {
        free(vp->execution_units[i]);
        vp->execution_units[i] = NULL;
    }
}

/**
 *
 */
int dague_fini( dague_context_t** pcontext )
{
    dague_context_t* context = *pcontext;
    int nb_total_comp_threads, p;

    /**
     * We need to force the main thread to drain all possible pending messages
     * on the communication layer. This is not an issue in a distributed run,
     * but on a single node run with MPI support, objects can be created (and
     * thus context_id additions might be pending on the communication layer).
     */
#if defined(DISTRIBUTED)
    if( (1 == dague_communication_engine_up) &&  /* engine enabled */
        (context->nb_nodes == 1) &&  /* single node: otherwise the messages will
                                      * be drained by the communication thread */
        DAGUE_THREAD_IS_MASTER(context->virtual_processes[0]->execution_units[0]) ) {
        /* check for remote deps completion */
        dague_remote_dep_progress(context->virtual_processes[0]->execution_units[0]);
    }
#endif /* defined(DISTRIBUTED) */

    /* Now wait until every thread is back */
    context->__dague_internal_finalization_in_progress = 1;
    dague_barrier_wait( &(context->barrier) );

    dague_rusage(true);

    PINS_THREAD_FINI(context->virtual_processes[0]->execution_units[0]);

    nb_total_comp_threads = 0;
    for(p = 0; p < context->nb_vp; p++) {
        nb_total_comp_threads += context->virtual_processes[p]->nb_cores;
    }

    /* The first execution unit is for the master thread */
    if( nb_total_comp_threads > 1 ) {
        for(p = 1; p < nb_total_comp_threads; p++) {
            pthread_join( context->pthreads[p], NULL );
        }
        free(context->pthreads);
        context->pthreads = NULL;
    }
    /* From now on all the thrteads have been shut-off, and they are supposed to
     * have cleaned all their provate memory. Unleash the global cleaning process.
     */

    PINS_FINI(context);

#ifdef DAGUE_PROF_TRACE
    dague_profiling_dbp_dump();
#endif  /* DAGUE_PROF_TRACE */

    (void) dague_remote_dep_fini(context);

    dague_remove_scheduler( context );

    dague_data_fini(context);

    dague_devices_remove(dague_device_cpus);
    free(dague_device_cpus);
    dague_device_cpus = NULL;

    dague_devices_remove(dague_device_recursive);
    free(dague_device_recursive);
    dague_device_recursive = NULL;

    dague_devices_fini(context);

    for(p = 0; p < context->nb_vp; p++) {
        dague_vp_fini(context->virtual_processes[p]);
        free(context->virtual_processes[p]);
        context->virtual_processes[p] = NULL;
    }

    AYU_FINI();
#ifdef DAGUE_PROF_TRACE
    (void)dague_profiling_fini( );  /* we're leaving, ignore errors */
#endif  /* DAGUE_PROF_TRACE */

    if(dague_enable_dot) {
#if defined(DAGUE_PROF_GRAPHER)
        dague_prof_grapher_fini();
#endif  /* defined(DAGUE_PROF_GRAPHER) */
        free(dague_enable_dot);
        dague_enable_dot = NULL;
    }
    /* Destroy all resources allocated for the barrier */
    dague_barrier_destroy( &(context->barrier) );

#if defined(DAGUE_HAVE_HWLOC_BITMAP)
    /* Release thread binding masks */
    hwloc_bitmap_free(context->cpuset_allowed_mask);
    hwloc_bitmap_free(context->cpuset_free_mask);

    dague_hwloc_fini();
#endif  /* DAGUE_HAVE_HWLOC_BITMAP */

    if (dague_app_name != NULL ) {
        free(dague_app_name);
        dague_app_name = NULL;
    }

#if defined(DAGUE_STATS)
    {
        char filename[64];
        char prefix[32];
#if defined(DISTRIBUTED) && defined(DAGUE_HAVE_MPI)
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        snprintf(filename, 64, "dague-%d.stats", rank);
        snprintf(prefix, 32, "%d/%d", rank, size);
# else
        snprintf(filename, 64, "dague.stats");
        prefix[0] = '\0';
# endif
        dague_stats_dump(filename, prefix);
    }
#endif

    dague_handle_empty_repository();

    dague_mca_param_finalize();
    dague_installdirs_close();
    dague_output_finalize();

    free(context);
    *pcontext = NULL;

    dague_class_finalize();
    dague_debug_fini();  /* Always last */
    return 0;
}

/**
 * Resolve all IN() dependencies for this particular instance of execution.
 */
static dague_dependency_t
dague_check_IN_dependencies_with_mask( const dague_handle_t *dague_handle,
                                       const dague_execution_context_t* exec_context )
{
    const dague_function_t* function = exec_context->function;
    int i, j, active;
    const dague_flow_t* flow;
    const dep_t* dep;
    dague_dependency_t ret = 0;

    if( !(function->flags & DAGUE_HAS_IN_IN_DEPENDENCIES) ) {
        return 0;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->in[i]); i++ ) {
        flow = function->in[i];

        /**
         * Controls and data have different logic:
         * Flows can depend conditionally on multiple input or control.
         * It is assumed that in the data case, one input will always become true.
         *  So, the Input dependency is already solved if one is found with a true cond,
         *      and depend only on the data.
         *
         * On the other hand, if all conditions for the control are false,
         * it is assumed that no control should be expected.
         */
        if( FLOW_ACCESS_NONE == (flow->flow_flags & FLOW_ACCESS_MASK) ) {
            active = (1 << flow->flow_index);
            /* Control case: resolved unless we find at least one input control */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( 0 == dep->cond->inline_func32(dague_handle, exec_context->locals) ) {
                        /* Cannot use control gather magic with the USE_DEPS_MASK */
                        assert( NULL == dep->ctl_gather_nb );
                        continue;
                    }
                }
                active = 0;
                break;
            }
        } else {
            if( !(flow->flow_flags & FLOW_HAS_IN_DEPS) ) continue;
            if( NULL == flow->dep_in[0] ) {
                /** As the flow is tagged with FLOW_HAS_IN_DEPS and there is no
                 * dep_in we are in the case where a write only dependency used
                 * an in dependency to specify the arena where the data should
                 * be allocated.
                 */
                active = (1 << flow->flow_index);
            } else {
                /* Data case: resolved only if we found a data already ready */
                for( active = 0, j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                    dep = flow->dep_in[j];
                    if( NULL != dep->cond ) {
                        /* Check if the condition apply on the current setting */
                        assert( dep->cond->op == EXPR_OP_INLINE );
                        if( 0 == dep->cond->inline_func32(dague_handle, exec_context->locals) )
                            continue;  /* doesn't match */
                        /* the condition triggered let's check if it's for a data */
                    }  /* otherwise we have an input flow without a condition, it MUST be final */
                    if( 0xFF == dep->function_id )
                        active = (1 << flow->flow_index);
                    break;
                }
            }
        }
        ret |= active;
    }
    return ret;
}

static dague_dependency_t
dague_check_IN_dependencies_with_counter( const dague_handle_t *dague_handle,
                                          const dague_execution_context_t* exec_context )
{
    const dague_function_t* function = exec_context->function;
    int i, j, active;
    const dague_flow_t* flow;
    const dep_t* dep;
    dague_dependency_t ret = 0;

    if( !(function->flags & DAGUE_HAS_CTL_GATHER) &&
        !(function->flags & DAGUE_HAS_IN_IN_DEPENDENCIES) ) {
        /* If the number of goal does not depend on this particular task instance,
         * it is pre-computed by the daguepp compiler
         */
        return function->dependencies_goal;
    }

    for( i = 0; (i < MAX_PARAM_COUNT) && (NULL != function->in[i]); i++ ) {
        flow = function->in[i];

        /**
         * Controls and data have different logic:
         * Flows can depend conditionally on multiple input or control.
         * It is assumed that in the data case, one input will always become true.
         *  So, the Input dependency is already solved if one is found with a true cond,
         *      and depend only on the data.
         *
         * On the other hand, if all conditions for the control are false,
         *  it is assumed that no control should be expected.
         */
        active = 0;
        if( FLOW_ACCESS_NONE == (flow->flow_flags & FLOW_ACCESS_MASK) ) {
            /* Control case: just count how many must be resolved */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( dep->cond->inline_func32(dague_handle, exec_context->locals) ) {
                        if( NULL == dep->ctl_gather_nb)
                            active++;
                        else {
                            assert( dep->ctl_gather_nb->op == EXPR_OP_INLINE );
                            active += dep->ctl_gather_nb->inline_func32(dague_handle, exec_context->locals);
                        }
                    }
                } else {
                    if( NULL == dep->ctl_gather_nb)
                        active++;
                    else {
                        assert( dep->ctl_gather_nb->op == EXPR_OP_INLINE );
                        active += dep->ctl_gather_nb->inline_func32(dague_handle, exec_context->locals);
                    }
                }
            }
        } else {
            /* Data case: we count how many inputs we must have (the opposite
             * compared with the mask case). We iterate over all the input
             * dependencies of the flow to make sure the flow is expected to
             * hold a valid value.
             */
            for( j = 0; (j < MAX_DEP_IN_COUNT) && (NULL != flow->dep_in[j]); j++ ) {
                dep = flow->dep_in[j];
                if( NULL != dep->cond ) {
                    /* Check if the condition apply on the current setting */
                    assert( dep->cond->op == EXPR_OP_INLINE );
                    if( 0 == dep->cond->inline_func32(dague_handle, exec_context->locals) )
                        continue;  /* doesn't match */
                    /* the condition triggered let's check if it's for a data */
                } else {
                    /* we have an input flow without a condition, it MUST be final */
                }
                if( 0xFF != dep->function_id )  /* if not a data we must wait for the flow activation */
                    active++;
                break;
            }
        }
        ret += active;
    }
    return ret;
}

dague_dependency_t *dague_default_find_deps(const dague_handle_t *dague_handle,
                                            const dague_execution_context_t* restrict exec_context)
{
    dague_dependencies_t *deps;
    int p;

    deps = dague_handle->dependencies_array[exec_context->function->function_id];
    assert( NULL != deps );

    for(p = 0; p < exec_context->function->nb_parameters - 1; p++) {
        assert( (deps->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) != 0 );
        deps = deps->u.next[exec_context->locals[exec_context->function->params[p]->context_index].value - deps->min];
        assert( NULL != deps );
    }

    return &(deps->u.dependencies[exec_context->locals[exec_context->function->params[p]->context_index].value - deps->min]);
}

static int dague_update_deps_with_counter(const dague_handle_t *dague_handle,
                                          const dague_execution_context_t* restrict exec_context,
                                          dague_dependency_t *deps)
{
    dague_dependency_t dep_new_value, dep_cur_value;
#if defined(DAGUE_DEBUG_PARANOID) || defined(DAGUE_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, exec_context);
#endif

    if( 0 == *deps ) {
        dep_new_value = dague_check_IN_dependencies_with_counter( dague_handle, exec_context ) - 1;
        if( dague_atomic_cas( deps, 0, dep_new_value ) == 1 )
            dep_cur_value = dep_new_value;
        else
            dep_cur_value = dague_atomic_dec_32b( deps );
    } else {
        dep_cur_value = dague_atomic_dec_32b( deps );
    }
    DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Activate counter dependency for %s leftover %d (excluding current)",
            tmp, dep_cur_value);

#if defined(DAGUE_DEBUG_PARANOID)
    {
        char wtmp[MAX_TASK_STRLEN];
        if( (uint32_t)dep_cur_value > (uint32_t)-128) {
            dague_abort("function %s as reached an improbable dependency count of %u",
                  wtmp, dep_cur_value );
        }

        DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Task %s has a current dependencies count of %d remaining. %s to go!",
               tmp, dep_cur_value,
               (dep_cur_value == 0) ? "Ready" : "Not ready");
    }
#endif /* DAGUE_DEBUG_PARANOID */

    return dep_cur_value == 0;
}

static int dague_update_deps_with_mask(const dague_handle_t *dague_handle,
                                       const dague_execution_context_t* restrict exec_context,
                                       dague_dependency_t *deps,
                                       const dague_execution_context_t* restrict origin,
                                       const dague_flow_t* restrict origin_flow,
                                       const dague_flow_t* restrict dest_flow)
{
    dague_dependency_t dep_new_value, dep_cur_value;
    const dague_function_t* function = exec_context->function;
#if defined(DAGUE_DEBUG_NOISIER) || defined(DAGUE_DEBUG_PARANOID)
    char tmpo[MAX_TASK_STRLEN], tmpt[MAX_TASK_STRLEN];
    dague_snprintf_execution_context(tmpo, MAX_TASK_STRLEN, origin);
    dague_snprintf_execution_context(tmpt, MAX_TASK_STRLEN, exec_context);
#endif

    DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Activate mask dep for %s:%s (current 0x%x now 0x%x goal 0x%x) from %s:%s",
            dest_flow->name, tmpt, *deps, (1 << dest_flow->flow_index), function->dependencies_goal,
            origin_flow->name, tmpo);
#if defined(DAGUE_DEBUG_PARANOID)
    if( (*deps) & (1 << dest_flow->flow_index) ) {
        dague_abort("Output dependencies 0x%x from %s (flow %s) activate an already existing dependency 0x%x on %s (flow %s)",
               dest_flow->flow_index, tmpo,
               origin_flow->name, *deps,
               tmpt, dest_flow->name );
    }
#else
    (void) origin; (void) origin_flow;
#endif

    assert( 0 == (*deps & (1 << dest_flow->flow_index)) );

    dep_new_value = DAGUE_DEPENDENCIES_IN_DONE | (1 << dest_flow->flow_index);
    /* Mark the dependencies and check if this particular instance can be executed */
    if( !(DAGUE_DEPENDENCIES_IN_DONE & (*deps)) ) {
        dep_new_value |= dague_check_IN_dependencies_with_mask( dague_handle, exec_context );
#if defined(DAGUE_DEBUG_NOISIER)
        if( dep_new_value != 0 ) {
            DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Activate IN dependencies with mask 0x%x", dep_new_value);
        }
#endif
    }

    dep_cur_value = dague_atomic_bor( deps, dep_new_value );

#if defined(DAGUE_DEBUG_PARANOID)
    if( (dep_cur_value & function->dependencies_goal) == function->dependencies_goal ) {
        int success;
        dague_dependency_t tmp_mask;
        tmp_mask = *deps;
        success = dague_atomic_cas( deps,
                                    tmp_mask, (tmp_mask | DAGUE_DEPENDENCIES_TASK_DONE) );
        if( !success || (tmp_mask & DAGUE_DEPENDENCIES_TASK_DONE) ) {
            dague_abort("Task %s scheduled twice (second time by %s)!!!",
                   tmpt, tmpo);
        }
    }
#endif  /* defined(DAGUE_DEBUG_PARANOID) */

    DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "Task %s has a current dependencies of 0x%x and a goal of 0x%x. %s to go!",
            tmpt, dep_cur_value, function->dependencies_goal,
            ((dep_cur_value & function->dependencies_goal) == function->dependencies_goal) ?
            "Ready" : "Not ready");
    return (dep_cur_value & function->dependencies_goal) == function->dependencies_goal;
}

/**
 * Mark the task as having all it's dependencies satisfied. This is not
 * necessarily required for the startup process, but it leaves traces such that
 * all executed tasks will show consistently (no difference between the startup
 * tasks and later tasks).
 */
void dague_dependencies_mark_task_as_startup(dague_execution_context_t* restrict exec_context)
{
    const dague_function_t* function = exec_context->function;
    dague_handle_t *dague_handle = exec_context->dague_handle;
    dague_dependency_t *deps = function->find_deps(dague_handle, exec_context);

    if( function->flags & DAGUE_USE_DEPS_MASK ) {
        *deps = DAGUE_DEPENDENCIES_STARTUP_TASK | function->dependencies_goal;
    } else {
        *deps = 0;
    }
}

/**
 * Release the OUT dependencies for a single instance of a task. No ranges are
 * supported and the task is supposed to be valid (no input/output tasks) and
 * local.
 */
int dague_release_local_OUT_dependencies(dague_execution_unit_t* eu_context,
                                         const dague_execution_context_t* restrict origin,
                                         const dague_flow_t* restrict origin_flow,
                                         const dague_execution_context_t* restrict exec_context,
                                         const dague_flow_t* restrict dest_flow,
                                         data_repo_entry_t* dest_repo_entry,
                                         dague_dep_data_description_t* data,
                                         dague_execution_context_t** pready_ring)
{
    const dague_function_t* function = exec_context->function;
    dague_dependency_t *deps;
    int completed;
#if defined(DAGUE_DEBUG_NOISIER)
    char tmp1[MAX_TASK_STRLEN], tmp2[MAX_TASK_STRLEN];
    dague_snprintf_execution_context(tmp1, MAX_TASK_STRLEN, exec_context);
#endif

    DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "Activate dependencies for %s flags = 0x%04x", tmp1, function->flags);
    deps = function->find_deps(origin->dague_handle, exec_context);

    if( function->flags & DAGUE_USE_DEPS_MASK ) {
        completed = dague_update_deps_with_mask(origin->dague_handle, exec_context, deps, origin, origin_flow, dest_flow);
    } else {
        completed = dague_update_deps_with_counter(origin->dague_handle, exec_context, deps);
    }

#if defined(DAGUE_PROF_GRAPHER)
    dague_prof_grapher_dep(origin, exec_context, completed, origin_flow, dest_flow);
#endif  /* defined(DAGUE_PROF_GRAPHER) */

    if( completed ) {

        /* This task is ready to be executed as all dependencies are solved.
         * Queue it into the ready_list passed as an argument.
         */
        {
            dague_execution_context_t *new_context = (dague_execution_context_t *) dague_thread_mempool_allocate(eu_context->context_mempool);
            DAGUE_COPY_EXECUTION_CONTEXT(new_context, exec_context);
            new_context->status = DAGUE_TASK_STATUS_NONE;
            AYU_ADD_TASK(new_context);

            DAGUE_DEBUG_VERBOSE(6, dague_debug_output,
                   "%s becomes ready from %s on thread %d:%d, with mask 0x%04x and priority %d",
                   tmp1,
                   dague_snprintf_execution_context(tmp2, MAX_TASK_STRLEN, origin),
                   eu_context->th_id, eu_context->virtual_process->vp_id,
                   *deps,
                   exec_context->priority);

            assert( dest_flow->flow_index <= new_context->function->nb_flows);
            memset( new_context->data, 0, sizeof(dague_data_pair_t) * new_context->function->nb_flows);
            /**
             * Save the data_repo and the pointer to the data for later use. This will prevent the
             * engine from atomically locking the hash table for at least one of the flow
             * for each execution context.
             */
            new_context->data[(int)dest_flow->flow_index].data_repo = dest_repo_entry;
            new_context->data[(int)dest_flow->flow_index].data_in   = origin->data[origin_flow->flow_index].data_out;
            (void)data;
            AYU_ADD_TASK_DEP(new_context, (int)dest_flow->flow_index);

            if(exec_context->function->flags & DAGUE_IMMEDIATE_TASK) {
                DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "  Task %s is immediate and will be executed ASAP", tmp1);
                __dague_execute(eu_context, new_context);
                __dague_complete_execution(eu_context, new_context);
#if 0 /* TODO */
                SET_HIGHEST_PRIORITY(new_context, dague_execution_context_priority_comparator);
                DAGUE_LIST_ITEM_SINGLETON(&(new_context->list_item));
                if( NULL != (*pimmediate_ring) ) {
                    (void)dague_list_item_ring_push( (dague_list_item_t*)(*pimmediate_ring), &new_context->list_item );
                }
                *pimmediate_ring = new_context;
#endif
            } else {
                *pready_ring = (dague_execution_context_t*)
                    dague_list_item_ring_push_sorted( (dague_list_item_t*)(*pready_ring),
                                                      &new_context->super.list_item,
                                                      dague_execution_context_priority_comparator );
            }
        }
    } else { /* Service not ready */
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "  => Service %s not yet ready", tmp1);
    }

    return 0;
}

dague_ontask_iterate_t
dague_release_dep_fct(dague_execution_unit_t *eu,
                      const dague_execution_context_t *newcontext,
                      const dague_execution_context_t *oldcontext,
                      const dep_t* dep,
                      dague_dep_data_description_t* data,
                      int src_rank, int dst_rank, int dst_vpid,
                      void *param)
{
    dague_release_dep_fct_arg_t *arg = (dague_release_dep_fct_arg_t *)param;
    const dague_flow_t* src_flow = dep->belongs_to;

    /**
     * Check that we don't forward a NULL data to someone else. This
     * can be done only on the src node, since the dst node can
     * check for datatypes without knowing the data yet.
     * By checking now, we allow for the data to be created any time bfore we
     * actually try to transfer it.
     */
    if( DAGUE_UNLIKELY((data->data == NULL) &&
                       (eu->virtual_process->dague_context->my_rank == src_rank) &&
                       ((dep->belongs_to->flow_flags & FLOW_ACCESS_MASK) != FLOW_ACCESS_NONE)) ) {
        char tmp1[MAX_TASK_STRLEN], tmp2[MAX_TASK_STRLEN];
        dague_abort("A NULL is forwarded\n"
                    "\tfrom: %s flow %s\n"
                    "\tto:   %s flow %s",
                    dague_snprintf_execution_context(tmp1, MAX_TASK_STRLEN, oldcontext), dep->belongs_to->name,
                    dague_snprintf_execution_context(tmp2, MAX_TASK_STRLEN, newcontext), dep->flow->name);
    }

#if defined(DISTRIBUTED)
    if( dst_rank != src_rank ) {

        assert( 0 == (arg->action_mask & DAGUE_ACTION_RECV_INIT_REMOTE_DEPS) );

        if( arg->action_mask & DAGUE_ACTION_SEND_INIT_REMOTE_DEPS ){
            struct remote_dep_output_param_s* output;
            int _array_pos, _array_mask;

#if !defined(DAGUE_DIST_COLLECTIVES)
            assert(src_rank == eu->virtual_process->dague_context->my_rank);
#endif
            _array_pos = dst_rank / (8 * sizeof(uint32_t));
            _array_mask = 1 << (dst_rank % (8 * sizeof(uint32_t)));
            DAGUE_ALLOCATE_REMOTE_DEPS_IF_NULL(arg->remote_deps, oldcontext, MAX_PARAM_COUNT);
            output = &arg->remote_deps->output[dep->dep_datatype_index];
            assert( (-1 == arg->remote_deps->root) || (arg->remote_deps->root == src_rank) );
            arg->remote_deps->root = src_rank;
            arg->remote_deps->outgoing_mask |= (1 << dep->dep_datatype_index);
            if( !(output->rank_bits[_array_pos] & _array_mask) ) {
                output->rank_bits[_array_pos] |= _array_mask;
                output->deps_mask |= (1 << dep->dep_index);
                if( 0 == output->count_bits ) {
                    output->data = *data;
                } else {
                    assert(output->data.data == data->data);
                }
                output->count_bits++;
                if(newcontext->priority > output->priority) {
                    output->priority = newcontext->priority;
                    if(newcontext->priority > arg->remote_deps->max_priority)
                        arg->remote_deps->max_priority = newcontext->priority;
                }
            }  /* otherwise the bit is already flipped, the peer is already part of the propagation. */
        }
    }
#else
    (void)src_rank;
    (void)data;
#endif

    if( (arg->action_mask & DAGUE_ACTION_RELEASE_LOCAL_DEPS) &&
        (eu->virtual_process->dague_context->my_rank == dst_rank) ) {
        /* Old condition */
        /* if( FLOW_ACCESS_NONE != (src_flow->flow_flags & FLOW_ACCESS_MASK) ) { */

        /* Copying data in data-repo if there is data .
         * We are doing this in order for dtd to be able to track control dependences.
         */
        if( oldcontext->data[src_flow->flow_index].data_out != NULL ) {
            arg->output_entry->data[src_flow->flow_index] = oldcontext->data[src_flow->flow_index].data_out;
            arg->output_usage++;
            /* BEWARE: This increment is required to be done here. As the target task
             * bits are marked, another thread can now enable the task. Once schedulable
             * the task will try to access its input data and decrement their ref count.
             * Thus, if the ref count is not increased here, the data might dissapear
             * before this task released it completely.
             */
            OBJ_RETAIN( arg->output_entry->data[src_flow->flow_index] );
        }
        dague_release_local_OUT_dependencies(eu, oldcontext, src_flow,
                                             newcontext, dep->flow,
                                             arg->output_entry,
                                             data,
                                             &arg->ready_lists[dst_vpid]);
    }

    return DAGUE_ITERATE_CONTINUE;
}

/**
 * Convert the execution context to a string.
 */
char* dague_snprintf_execution_context( char* str, size_t size,
                                        const dague_execution_context_t* task)
{
    const dague_function_t* function = task->function;
    unsigned int i, ip, index = 0, is_param;

    assert( NULL != task->dague_handle );
    index += snprintf( str + index, size - index, "%s(", function->name );
    if( index >= size ) return str;
    for( ip = 0; ip < function->nb_parameters; ip++ ) {
        index += snprintf( str + index, size - index, "%s%d",
                           (ip == 0) ? "" : ", ",
                           task->locals[function->params[ip]->context_index].value );
        if( index >= size ) return str;
    }
    index += snprintf(str + index, size - index, ")[");
    if( index >= size ) return str;

    for( i = 0; i < function->nb_locals; i++ ) {
        is_param = 0;
        for( ip = 0; ip < function->nb_parameters; ip++ ) {
            if(function->params[ip]->context_index == function->locals[i]->context_index) {
                is_param = 1;
                break;
            }
        }
        index += snprintf( str + index, size - index,
                           (is_param ? "%s%d" : "[%s%d]"),
                           (i == 0) ? "" : ", ",
                           task->locals[i].value );
        if( index >= size ) return str;
    }
    index += snprintf(str + index, size - index, "]<%d>{%u}", task->priority, task->dague_handle->handle_id );

    return str;
}
/**
 * Convert assignments to a string.
 */
char* dague_snprintf_assignments( char* str, size_t size,
                                  const dague_function_t* function,
                                  const assignment_t* locals)
{
    unsigned int ip, index = 0;

    index += snprintf( str + index, size - index, "%s", function->name );
    if( index >= size ) return str;
    for( ip = 0; ip < function->nb_parameters; ip++ ) {
        index += snprintf( str + index, size - index, "%s%d",
                           (ip == 0) ? "(" : ", ",
                           locals[function->params[ip]->context_index].value );
        if( index >= size ) return str;
    }
    index += snprintf(str + index, size - index, ")" );

    return str;
}


void dague_destruct_dependencies(dague_dependencies_t* d)
{
    int i;
    if( (d != NULL) && (d->flags & DAGUE_DEPENDENCIES_FLAG_NEXT) ) {
        for(i = d->min; i <= d->max; i++)
            if( NULL != d->u.next[i - d->min] )
                dague_destruct_dependencies(d->u.next[i-d->min]);
    }
    free(d);
}

/**
 *
 */
int dague_set_complete_callback( dague_handle_t* dague_handle,
                                 dague_event_cb_t complete_cb, void* complete_cb_data )
{
    if( NULL == dague_handle->on_complete ) {
        dague_handle->on_complete      = complete_cb;
        dague_handle->on_complete_data = complete_cb_data;
        return 0;
    }
    return -1;
}

/**
 *
 */
int dague_get_complete_callback( const dague_handle_t* dague_handle,
                                 dague_event_cb_t* complete_cb, void** complete_cb_data )
{
    if( NULL != dague_handle->on_complete ) {
        *complete_cb      = dague_handle->on_complete;
        *complete_cb_data = dague_handle->on_complete_data;
        return 0;
    }
    return -1;
}

/**
 *
 */
int dague_set_enqueue_callback( dague_handle_t* dague_handle,
                                dague_event_cb_t enqueue_cb, void* enqueue_cb_data )
{
    if( NULL == dague_handle->on_enqueue ) {
        dague_handle->on_enqueue      = enqueue_cb;
        dague_handle->on_enqueue_data = enqueue_cb_data;
        return 0;
    }
    return -1;
}

/**
 *
 */
int dague_get_enqueue_callback( const dague_handle_t* dague_handle,
                                dague_event_cb_t* enqueue_cb, void** enqueue_cb_data )
{
    if( NULL != dague_handle->on_enqueue ) {
        *enqueue_cb      = dague_handle->on_enqueue;
        *enqueue_cb_data = dague_handle->on_enqueue_data;
        return 0;
    }
    return -1;
}

/* TODO: Change this code to something better */
static volatile uint32_t object_array_lock = 0;
static dague_handle_t** object_array = NULL;
static uint32_t object_array_size = 1, object_array_pos = 0;
#define NOOBJECT ((void*)-1)

static void dague_handle_empty_repository(void)
{
    dague_atomic_lock( &object_array_lock );
    free(object_array);
    object_array = NULL;
    object_array_size = 1;
    object_array_pos = 0;
    dague_atomic_unlock( &object_array_lock );
}

/**< Retrieve the local object attached to a unique object id */
dague_handle_t* dague_handle_lookup( uint32_t handle_id )
{
    dague_handle_t *r = NOOBJECT;
    dague_atomic_lock( &object_array_lock );
    if( handle_id <= object_array_pos ) {
        r = object_array[handle_id];
    }
    dague_atomic_unlock( &object_array_lock );
    return (NOOBJECT == r ? NULL : r);
}

/**< Reverse an unique ID for the handle but without adding the object to the management array.
 *   Beware that on a distributed environment the connected objects must have the same ID.
 */
int dague_handle_reserve_id( dague_handle_t* object )
{
    uint32_t idx;

    dague_atomic_lock( &object_array_lock );
    idx = (uint32_t)++object_array_pos;

    if( (NULL == object_array) || (idx >= object_array_size) ) {
        object_array_size <<= 1;
        object_array = (dague_handle_t**)realloc(object_array, object_array_size * sizeof(dague_handle_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = (object_array_size>>1); i < object_array_size;
             object_array[i++] = NOOBJECT );
    }
    object->handle_id = idx;
    assert( NOOBJECT == object_array[idx] );
    dague_atomic_unlock( &object_array_lock );
    return idx;
}

/**< Register a handle object with the engine. Once enrolled the object can be target
 * for other components of the runtime, such as communications.
 */
int dague_handle_register( dague_handle_t* object)
{
    uint32_t idx = object->handle_id;

    dague_atomic_lock( &object_array_lock );
    if( (NULL == object_array) || (idx >= object_array_size) ) {
        object_array_size <<= 1;
        object_array = (dague_handle_t**)realloc(object_array, object_array_size * sizeof(dague_handle_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = (object_array_size>>1); i < object_array_size;
             object_array[i++] = NOOBJECT );
    }
    object_array[idx] = object;
    dague_atomic_unlock( &object_array_lock );
    return idx;
}

/**< globally synchronize object id's so that next register generates the same
 * id at all ranks. */
void dague_handle_sync_ids( void )
{
    uint32_t idx;
    dague_atomic_lock( &object_array_lock );
    idx = (int)object_array_pos;
#if defined(DISTRIBUTED) && defined(DAGUE_HAVE_MPI)
    MPI_Allreduce( MPI_IN_PLACE, &idx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
#endif
    if( idx >= object_array_size ) {
        object_array_size <<= 1;
        object_array = (dague_handle_t**)realloc(object_array, object_array_size * sizeof(dague_handle_t*) );
        /* NULLify all the new elements */
        for( uint32_t i = (object_array_size>>1); i < object_array_size;
             object_array[i++] = NOOBJECT );
    }
    object_array_pos = idx;
    dague_atomic_unlock( &object_array_lock );
}

/**< Unregister the object with the engine. This make the handle available for
 * future handles. Beware that in a distributed environment the connected objects
 * must have the same ID.
 */
void dague_handle_unregister( dague_handle_t* object )
{
    dague_atomic_lock( &object_array_lock );
    assert( object->handle_id < object_array_size );
    assert( object_array[object->handle_id] == object );
    assert( DAGUE_RUNTIME_RESERVED_NB_TASKS == object->nb_tasks );
    assert( 0 == object->nb_pending_actions );
    object_array[object->handle_id] = NOOBJECT;
    dague_atomic_unlock( &object_array_lock );
}

void dague_handle_free(dague_handle_t *handle)
{
    if( NULL == handle )
        return;
    if( NULL == handle->destructor ) {
        free( handle );
        return;
    }
    /* the destructor calls the appropriate free on the handle */
    handle->destructor( handle );
}

/**
 * The final step of a handle activation. At this point we assume that all the local
 * initializations have been successfully completed for all components, and that the
 * handle is ready to be registered with the system, and any potential pending tasks
 * ready to go. If distributed is non 0, then the runtime assumes that the handle has
 * a distributed scope and should be registered with the communication engine.
 *
 * The local_task allows for concurrent management of the startup_queue, and provide a way
 * to prevent a task from being added to the scheduler. As the different tasks classes are
 * initialized concurrently, we need a way to prevent the beginning of the tasks generation until
 * all the tasks classes associated with a DAG are completed. Thus, until the synchronization
 * is complete, the task generators are put on hold in the startup_queue. Once the handle is
 * ready to advance, and this is the same moment as when the handle is ready to be enabled,
 * we reactivate all pending tasks, starting the tasks generation step for all type classes.
 */
int dague_handle_enable(dague_handle_t* handle,
                        dague_execution_context_t** startup_queue,
                        dague_execution_context_t* local_task,
                        dague_execution_unit_t * eu,
                        int distributed)
{
    if( NULL != startup_queue ) {
        dague_list_item_t *ring = NULL;
        dague_execution_context_t* ttask = (dague_execution_context_t*)*startup_queue;

        while( NULL != (ttask = (dague_execution_context_t*)*startup_queue) ) {
            /* Transform the single linked list into a ring */
            *startup_queue = (dague_execution_context_t*)ttask->super.list_item.list_next;
            if(ttask != local_task) {
                ttask->status = DAGUE_TASK_STATUS_HOOK;
                DAGUE_LIST_ITEM_SINGLETON(ttask);
                if(NULL == ring) ring = (dague_list_item_t *)ttask;
                else dague_list_item_ring_push(ring, &ttask->super.list_item);
            }
        }
        if( NULL != ring ) __dague_schedule(eu, (dague_execution_context_t *)ring);
    }
    /* Always register the handle. This allows the handle_t destructor to unregister it in all cases. */
    dague_handle_register(handle);
    if( 0 != distributed ) {
        (void)dague_remote_dep_new_object(handle);
    }
    (void)eu;
    return DAGUE_HOOK_RETURN_DONE;
}

/**< Print DAGuE usage message */
void dague_usage(void)
{
    dague_output(0,"\n"
            "A DAGuE argument sequence prefixed by \"--\" can end the command line\n\n"
            "     --dague_bind_comm   : define the core the communication thread will be bound on\n"
            "\n"
            "     Warning:: The binding options rely on hwloc. The core numerotation is defined between 0 and the number of cores.\n"
            "     Be careful when used with cgroups.\n"
            "\n"
            "    --help         : this message\n"
            "\n"
            " -c --cores        : number of concurent threads (default: number of physical hyper-threads)\n"
            " -g --gpus         : number of GPU (default: 0)\n"
            " -o --scheduler    : select the scheduler (default: LFQ)\n"
            "                     Accepted values:\n"
            "                       LFQ -- Local Flat Queues\n"
            "                       GD  -- Global Dequeue\n"
            "                       LHQ -- Local Hierarchical Queues\n"
            "                       AP  -- Absolute Priorities\n"
            "                       PBQ -- Priority Based Local Flat Queues\n"
            "                       LTQ -- Local Tree Queues\n"
            "\n"
            "    --dot[=file]   : create a dot output file (default: don't)\n"
            "\n"
            "    --ht nbth      : enable a SMT/HyperThreadind binding using nbth hyper-thread per core.\n"
            "                     This parameter must be declared before the virtual process distribution parameter\n"
            " -V --vpmap        : select the virtual process map (default: flat map)\n"
            "                     Accepted values:\n"
            "                       flat  -- Flat Map: all cores defined with -c are under the same virtual process\n"
            "                       hwloc -- Hardware Locality based: threads up to -c are created and threads\n"
            "                                bound on cores that are under the same socket are also under the same\n"
            "                                virtual process\n"
            "                       rr:n:p:c -- create n virtual processes per real process, each virtual process with p threads\n"
            "                                   bound in a round-robin fashion on the number of cores c (overloads the -c flag)\n"
            "                       file:filename -- uses filename to load the virtual process map. Each entry details a virtual\n"
            "                                        process mapping using the semantic  [mpi_rank]:nb_thread:binding  with:\n"
            "                                        - mpi_rank : the mpi process rank (empty if not relevant)\n"
            "                                        - nb_thread : the number of threads under the virtual process\n"
            "                                                      (overloads the -c flag)\n"
            "                                        - binding : a set of cores for the thread binding. Accepted values are:\n"
            "                                          -- a core list          (exp: 1,3,5-6)\n"
            "                                          -- a hexadecimal mask   (exp: 0xff012)\n"
            "                                          -- a binding range expression: [start];[end];[step] \n"
            "                                             wich defines a round-robin one thread per core distribution from start\n"
            "                                             (default 0) to end (default physical core number) by step (default 1)\n"
            "\n"
            );
}




/* Parse --dague_bind parameter (define a set of cores for the thread binding)
 * The parameter can be
 * - a file containing the parameters (list, mask or expression) for each processes
 * - or a comma separated list of
 *   - a core
 *   - a hexadecimal mask
 *   - a range expression (a:[b[:c]])
 *
 * The function rely on a version of hwloc which support for bitmap.
 * It redefines the fields "bindto" of the startup structure used to initialize the threads
 *
 * We use the topology core indexes to define the binding, not the core numbers.
 * The index upper/lower bounds are 0 and (number_of_cores - 1).
 * The core_index_mask stores core indexes and will be converted into a core_number_mask
 * for the hwloc binding.
 */

#if defined(DAGUE_HAVE_HWLOC) && defined(DAGUE_HAVE_HWLOC_BITMAP)
#define DAGUE_BIND_THREAD(THR, WHERE)                                   \
    do {                                                                \
        int __where = (WHERE);                                          \
        if( (THR) < nb_total_comp_threads ) {                           \
            startup[(THR)].bindto = __where;  /* set the thread binding if legit */ \
            (THR)++;                                                    \
            if( hwloc_bitmap_isset(context->cpuset_allowed_mask, __where) ) { \
                dague_warning("Oversubscription on core %d detected\n", __where); \
            }                                                           \
        }                                                               \
        hwloc_bitmap_set(context->cpuset_allowed_mask, __where);  /* update the mask */ \
    } while (0)
#endif  /* defined(DAGUE_HAVE_HWLOC) && defined(DAGUE_HAVE_HWLOC_BITMAP) */

int dague_parse_binding_parameter(void * optarg, dague_context_t* context,
                                  __dague_temporary_thread_initialization_t* startup)
{
#if defined(DAGUE_HAVE_HWLOC) && defined(DAGUE_HAVE_HWLOC_BITMAP)
    char *option = optarg, *position, *endptr;
    int i, thr_idx = 0, nb_total_comp_threads = 0, where;
    int nb_real_cores = dague_hwloc_nb_real_cores();

    if( NULL == context->cpuset_allowed_mask )
        context->cpuset_allowed_mask = hwloc_bitmap_alloc();

    for(i = 0; i < context->nb_vp; i++)
        nb_total_comp_threads += context->virtual_processes[i]->nb_cores;
    if( NULL == optarg ) {
        for( thr_idx = 0; thr_idx < nb_total_comp_threads; ) {
            DAGUE_BIND_THREAD(thr_idx, (thr_idx % nb_real_cores));
        }
        if( nb_total_comp_threads < nb_real_cores )
            hwloc_bitmap_set_range(context->cpuset_allowed_mask, nb_total_comp_threads, nb_real_cores-1);
        goto compute_free_mask;
    }
    /* The parameter is a file */
    if( NULL != (position = strstr(option, "file:")) ) {
        /* Read from the file the binding parameter set for the local process and parse it
         (recursive call). */

        char *filename = position + strlen("file:");
        FILE *f;
        char *line = NULL;
        size_t line_len = 0;

        f = fopen(filename, "r");
        if( NULL == f ) {
            dague_warning("invalid binding file %s.", filename);
            return -1;
        }

        int rank = 0, line_num = 0;
#if defined(DISTRIBUTED) && defined(DAGUE_HAVE_MPI)
        /* distributed version: set the rank to find the corresponding line in the rankfile */
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif /* DISTRIBUTED && DAGUE_HAVE_MPI */
        while (getline(&line, &line_len, f) != -1) {
            if(line_num == rank) {
                DAGUE_DEBUG_VERBOSE(10, dague_debug_output, "MPI_process %i uses the binding parameters: %s", rank, line);
                dague_parse_binding_parameter(line, context, startup);
                break;
            }
            line_num++;
        }
        if( NULL != line )
            free(line);

        fclose(f);
        return -1;
    }

    if( (option[0] == '+') && (context->comm_th_core == -1)) {
        /* The parameter starts with "+" and no specific binding is (yet) defined
         * for the communication thread. The communication thread is then included
         * in the thread mapping. */
        context->comm_th_core = -2;
        option++;  /* skip the + */
    }

    if( NULL == context->cpuset_allowed_mask )
        context->cpuset_allowed_mask = hwloc_bitmap_alloc();
    /* From now on the option is a comma separated list of entities that can be
     * either single numbers, hexadecimal masks or [::] ranges with steps.
     */
    while( NULL != option ) {
        if( NULL != (position = strchr(option, 'x')) ) {
            option = position + 1;  /* skip the x */
            /* find the end of the hexa mask and parse it in reverse */
            position = strchr(option, ',');
            if( NULL == position )  /* we reached the end of the string, the last char is the one right in front */
                position = option + strlen(option) - 1;
            where = 0;
            while( position != option ) {
                long int mask = strtol(position, &endptr, 16);
                if( (0 == mask) && (position == endptr) ) {
                    dague_warning("binding: invalid char (%c) in hexadecimal mask. skip\n", position);
                    goto next_iteration;
                }
                assert( (0 <= mask) && (mask < 16) );  /* must be single hexa digit */
                for( i = 0; i < 4; i++ ) {
                    if( mask & (1<<i) ) {  /* bit is set */
                        DAGUE_BIND_THREAD(thr_idx, where);
                    }
                    where++;
                }
                position--;  /* reverse parsing to maintain the natural order of bits */
            }
            goto next_iteration;
        }

        if( NULL != (position = strchr(option, ':'))) {
            /* The parameter is a range expression such as [start]:[end]:[step] */
            int start = 0, step, end = nb_real_cores;
            if( position != option ) {
                /* we have a starting position */
                start = strtol(option, NULL, 10);
                if( (start >= nb_real_cores) || (start < 0) ) {
                    start = 0;
                    dague_warning("binding start core not valid (restored to %d)", start);
                }
            }
            position++;  /* skip the : */
            if( '\0' != position[0] ) {
                /* check for the ending position */
                if( ':' != position[0] ) {
                    end = strtol(position, &position, 10);
                    if( (end >= nb_real_cores) || (end < 0) ) {
                        end = nb_real_cores;
                        dague_warning("binding end core not valid (restored to default %d)", end);
                    }
                }
                position = strchr(position, ':');  /* find the step */
            }
            step = (start < end ? 1 : -1);
            if( NULL != position ) {
                position++;  /* skip the : directly into the step */
                if( '\0' != position[0] ) {
                    step = strtol(position, &endptr, 10); /* allow all numbers but 0 */
                    if( (0 == step) && (position == endptr) ) {
                        step = (start < end ? 1 : -1);
                    }
                }
            }
            if( (0 == step) || ((step > 0) && (start > end)) || ((step < 0) && (start < end)) ) {
                dague_warning("user provided binding step (%d) invalid. corrected\n", step);
                step = (start < end ? 1 : -1);
            }
            DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "binding defined by core range [%d:%d:%d]",
                                start, end, step);

            /* redefine the core according to the trio start/end/step */
            where = start;
            while( ((step > 0) && (where <= end)) || ((step < 0) && (where >= end)) ) {
                DAGUE_BIND_THREAD(thr_idx, where);
                where += step;
            }
        }

        else {  /* List of cores */
            where = strtol(option, &option, 10);
            if( !((where < nb_real_cores) && (where > -1)) ) {
                dague_warning("binding core #%i not valid (must be between 0 and %i (nb_core-1)\n",
                              where, nb_real_cores-1);
                goto next_iteration;
            }
            DAGUE_BIND_THREAD(thr_idx, where);
        }
      next_iteration:
        option = strchr(option, ',');  /* skip to the next comma */
        if( NULL != option ) option++;
    }
    /* All not-bounded threads will be unleashed */
    for( ; thr_idx < nb_total_comp_threads; thr_idx++ )
        startup[thr_idx].bindto = -1;

  compute_free_mask:
    /**
     * Compute the cpuset_free_mask bitmap, by excluding all the cores with
     * bound threads from the cpuset_allowed_mask.
     */
    context->cpuset_free_mask = hwloc_bitmap_dup(context->cpuset_allowed_mask);
    /* update the cpuset_free_mask according to the thread binding defined */
    for(thr_idx = 0; thr_idx < nb_total_comp_threads; thr_idx++)
        if( -1 != startup[thr_idx].bindto )
            hwloc_bitmap_clr(context->cpuset_free_mask, startup[thr_idx].bindto);

#if defined(DAGUE_DEBUG_NOISIER)
    {
        char *str = NULL;
        hwloc_bitmap_asprintf(&str, context->cpuset_allowed_mask);
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,
                            "Thread binding: cpuset [ALLOWED  ]: %s", str);
        free(str);
        hwloc_bitmap_asprintf(&str, context->cpuset_free_mask);
        DAGUE_DEBUG_VERBOSE(10, dague_debug_output,
                            "Thread binding: cpuset [AVAILABLE]: %s", str);
        free(str);
    }
#endif  /* defined(DAGUE_DEBUG_NOISIER) */

    return 0;
#else
    (void)optarg;
    (void)context;
    (void)startup;
    dague_warning("the binding defined by --dague_bind has been ignored (requires a build with HWLOC with bitmap support).");
    return -1;
#endif /* DAGUE_HAVE_HWLOC && DAGUE_HAVE_HWLOC_BITMAP */
}

static int dague_parse_comm_binding_parameter(void * optarg, dague_context_t* context)
{
#if defined(DAGUE_HAVE_HWLOC)
    char* option = optarg;
    if( option[0]!='\0' ) {
        int core = atoi(optarg);
        if( (core > -1) && (core < dague_hwloc_nb_real_cores()) )
            context->comm_th_core = core;
        else
            dague_warning("the binding defined by --dague_bind_comm has been ignored (illegal core number)");
    } else {
        /* TODO:: Add binding NUIOA aware by default */
        DAGUE_DEBUG_VERBOSE(20, dague_debug_output, "default binding for the communication thread");
    }
    return 0;
#else
    (void)optarg; (void)context;
    dague_warning("The binding defined by --dague_bind_comm has been ignored (requires HWLOC use with bitmap support).");
    return -1;
#endif  /* DAGUE_HAVE_HWLOC */
}

#if defined(DAGUE_SIM)
int dague_getsimulationdate( dague_context_t *dague_context ){
    return dague_context->largest_simulation_date;
}
#endif

static int32_t dague_expr_eval32(const expr_t *expr, dague_execution_context_t *context)
{
    dague_handle_t *handle = context->dague_handle;

    assert( expr->op == EXPR_OP_INLINE );
    return expr->inline_func32(handle, context->locals);
}

static int dague_debug_enumerate_next_in_execution_space(dague_execution_context_t *context,
                                                         int init, int li)
{
    const dague_function_t *function = context->function;
    int cur, max, incr, min;

    if( li == function->nb_locals )
        return init; /** We did not find a new context */

    min = dague_expr_eval32(function->locals[li]->min,
                            context);

    max = dague_expr_eval32(function->locals[li]->max,
                            context);
    if ( min > max ) {
        return 0; /* There is no context starting with these locals */
    }

    if( init ) {
        context->locals[li].value = min;
    }

    do {
        if( dague_debug_enumerate_next_in_execution_space(context, init, li+1) )
            return 1; /** We did find a new context */

        if( min == max )
            return 0; /** We can't change this local */

        cur = context->locals[li].value;
        if( function->locals[li]->expr_inc == NULL ) {
            incr = function->locals[li]->cst_inc;
        } else {
            incr = dague_expr_eval32(function->locals[li]->expr_inc, context);
        }

        if( cur + incr > max ) {
            return 0;
        }
        context->locals[li].value = cur + incr;
        init = 1;
    } while(1);
}

void dague_debug_print_local_expecting_tasks_for_function( dague_handle_t *handle,
                                                           const dague_function_t *function,
                                                           int show_remote,
                                                           int show_startup,
                                                           int show_complete,
                                                           int *nlocal,
                                                           int *nreleased,
                                                           int *ntotal)
{
    dague_execution_context_t context;
    dague_dependency_t *dep;
    dague_data_ref_t ref;
    int li, init;

    DAGUE_LIST_ITEM_SINGLETON( &context.super.list_item );
    context.super.mempool_owner = NULL;
    context.dague_handle = handle;
    context.function = function;
    context.priority = -1;
    context.status = DAGUE_TASK_STATUS_NONE;
    memset( context.data, 0, MAX_PARAM_COUNT * sizeof(dague_data_pair_t) );

    *nlocal = 0;
    *nreleased = 0;
    *ntotal = 0;

    /* For debugging purposes */
    for(li = 0; li < MAX_LOCAL_COUNT; li++) {
        context.locals[li].value = -1;
    }

    init = 1;
    while( dague_debug_enumerate_next_in_execution_space(&context, init, 0) ) {
        char tmp[MAX_TASK_STRLEN];
        init = 0;

        (*ntotal)++;
        function->data_affinity(&context, &ref);
        if( ref.ddesc->rank_of_key(ref.ddesc, ref.key) == ref.ddesc->myrank ) {
            (*nlocal)++;
            dep = function->find_deps(handle, &context);
            if( function->flags & DAGUE_USE_DEPS_MASK ) {
                if( *dep & DAGUE_DEPENDENCIES_STARTUP_TASK ) {
                    (*nreleased)++;
                    if( show_startup )
                        dague_debug_verbose(0, dague_debug_output, "  Task %s is a local startup task",
                                            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context));
                } else {
                    if((*dep & DAGUE_DEPENDENCIES_BITMASK) == function->dependencies_goal) {
                        (*nreleased)++;
                    }
                    if( show_complete ||
                        ((*dep & DAGUE_DEPENDENCIES_BITMASK) != function->dependencies_goal) ) {
                        dague_debug_verbose(0, dague_debug_output, "  Task %s is a local task with dependency 0x%08x (goal is 0x%08x) -- Flags: %s %s",
                                            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context),
                                            *dep & DAGUE_DEPENDENCIES_BITMASK,
                                            function->dependencies_goal,
                                            *dep & DAGUE_DEPENDENCIES_TASK_DONE ? "TASK_DONE" : "",
                                            *dep & DAGUE_DEPENDENCIES_IN_DONE ? "IN_DONE" : "");
                    }
                }
            } else {
                if( *dep == 0 )
                    (*nreleased)++;

                if( (*dep != 0) || show_complete )
                    dague_debug_verbose(0, dague_debug_output, "  Task %s is a local task that must wait for %d more dependencies to complete -- using count method for this task (CTL gather)",
                                        dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context),
                                        *dep);
            }
        } else {
            if( show_remote )
                dague_debug_verbose(0, dague_debug_output, "  Task %s is a remote task",
                                    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, &context));
        }
    }
}

void dague_debug_print_local_expecting_tasks_for_handle( dague_handle_t *handle,
                                                         int show_remote, int show_startup, int show_complete)
{
    uint32_t fi;
    int nlocal, ntotal, nreleased;
    /* The handle has not been initialized yet, or it has been completed */
    if( handle->dependencies_array == NULL )
        return;

    for(fi = 0; fi < handle->nb_functions; fi++) {
        dague_debug_verbose(0, dague_debug_output, " Tasks of Function %u (%s):\n", fi, handle->functions_array[fi]->name);
        dague_debug_print_local_expecting_tasks_for_function( handle, handle->functions_array[fi],
                                                              show_remote, show_startup, show_complete,
                                                              &nlocal, &nreleased, &ntotal );
        dague_debug_verbose(0, dague_debug_output, " Total number of Tasks of Class %s: %d\n", handle->functions_array[fi]->name, ntotal);
        dague_debug_verbose(0, dague_debug_output, " Local number of Tasks of Class %s: %d\n", handle->functions_array[fi]->name, nlocal);
        dague_debug_verbose(0, dague_debug_output, " Number of Tasks of Class %s that have been released: %d\n", handle->functions_array[fi]->name, nreleased);
    }
}

void dague_debug_print_local_expecting_tasks( int show_remote, int show_startup, int show_complete )
{
    dague_handle_t *handle;
    uint32_t oi;

    dague_atomic_lock( &object_array_lock );
    for( oi = 1; oi <= object_array_pos; oi++) {
        handle = object_array[ oi ];
        if( handle == NOOBJECT )
            continue;
        if( handle == NULL )
            continue;
        dague_debug_verbose(0, dague_debug_output, "Tasks of Handle %u:\n", oi);
        dague_debug_print_local_expecting_tasks_for_handle( handle,
                                                            show_remote,
                                                            show_startup,
                                                            show_complete );
    }
    dague_atomic_unlock( &object_array_lock );
}

/** deps is an array of size MAX_PARAM_COUNT
 *  Returns the number of output deps on which there is a final output
 */
int dague_task_deps_with_final_output(const dague_execution_context_t *task,
                                      const dep_t **deps)
{
    const dague_function_t *f;
    const dague_flow_t  *flow;
    const dep_t          *dep;
    int fi, di, nbout = 0;

    f = task->function;
    for(fi = 0; fi < f->nb_flows && f->out[fi] != NULL; fi++) {
        flow = f->out[fi];
        if( ! (SYM_OUT & flow->sym_type ) )
            continue;
        for(di = 0; di < MAX_DEP_OUT_COUNT && flow->dep_out[di] != NULL; di++) {
            dep = flow->dep_out[di];
            if( dep->function_id != (uint8_t)-1 )
                continue;
            if( NULL != dep->cond ) {
                assert( EXPR_OP_INLINE == dep->cond->op );
                if( dep->cond->inline_func32(task->dague_handle, task->locals) )
                    continue;
            }
            deps[nbout] = dep;
            nbout++;
        }
    }

    return nbout;
}
