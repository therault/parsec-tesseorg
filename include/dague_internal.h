/*
 * Copyright (c) 2012     The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_INTERNAL_H_HAS_BEEN_INCLUDED
#define DAGUE_INTERNAL_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "list_item.h"
#include "dague_description_structures.h"
#include "dague.h"

typedef struct dague_remote_deps_t dague_remote_deps_t;
typedef struct dague_arena_t dague_arena_t;
typedef struct dague_arena_chunk_t dague_arena_chunk_t;
typedef struct dague_data_pair_t dague_data_pair_t;

#ifdef HAVE_PAPI
#define MAX_EVENTS 3
#endif

/* There is another loop after this one. */
#define DAGUE_DEPENDENCIES_FLAG_NEXT       0x01
/* This is the final loop */
#define DAGUE_DEPENDENCIES_FLAG_FINAL      0x02
/* This loops array is allocated */
#define DAGUE_DEPENDENCIES_FLAG_ALLOCATED  0x04

/* The first time the IN dependencies are
 *       checked leave a trace in order to avoid doing it again.
 */
#define DAGUE_DEPENDENCIES_TASK_DONE      ((dague_dependency_t)(1<<31))
#define DAGUE_DEPENDENCIES_IN_DONE        ((dague_dependency_t)(1<<30))
#define DAGUE_DEPENDENCIES_BITMASK        (~(DAGUE_DEPENDENCIES_TASK_DONE|DAGUE_DEPENDENCIES_IN_DONE))

typedef union {
    dague_dependency_t    dependencies[1];
    dague_dependencies_t* next[1];
} dague_dependencies_union_t;

struct dague_dependencies_t {
    int                     flags;
    const symbol_t*         symbol;
    int                     min;
    int                     max;
    dague_dependencies_t* prev;
    /* keep this as the last field in the structure */
    dague_dependencies_union_t u;
};

void dague_destruct_dependencies(dague_dependencies_t* d);

/**
 * Functions for DAG manipulation.
 */
typedef enum  {
    DAGUE_ITERATE_STOP,
    DAGUE_ITERATE_CONTINUE
} dague_ontask_iterate_t;

typedef int (dague_hook_t)(struct dague_execution_unit*, dague_execution_context_t*);
typedef int (dague_release_deps_t)(struct dague_execution_unit*,
                                   dague_execution_context_t*,
                                   uint32_t,
                                   struct dague_remote_deps_t *);

typedef dague_ontask_iterate_t (dague_ontask_function_t)(struct dague_execution_unit *eu,
                                                         dague_execution_context_t *newcontext,
                                                         dague_execution_context_t *oldcontext,
                                                         int flow_index, int outdep_index,
                                                         int rank_src, int rank_dst,
                                                         int vpid_dst,
                                                         dague_arena_t* arena,
                                                         int nb_elt,
                                                         void *param);
typedef void (dague_traverse_function_t)(struct dague_execution_unit *,
                                         dague_execution_context_t *,
                                         uint32_t,
                                         dague_ontask_function_t *,
                                         void *);

#if defined(DAGUE_SIM)
typedef int (dague_sim_cost_fct_t)(const dague_execution_context_t *exec_context);
#endif
typedef uint64_t (dague_functionkey_fn_t)(const dague_object_t *dague_object, const assignment_t *assignments);

#define DAGUE_HAS_IN_IN_DEPENDENCIES     0x0001
#define DAGUE_HAS_OUT_OUT_DEPENDENCIES   0x0002
#define DAGUE_HAS_IN_STRONG_DEPENDENCIES 0x0004
#define DAGUE_HIGH_PRIORITY_TASK         0x0008
#define DAGUE_IMMEDIATE_TASK             0x0010

struct dague_function {
    const char                  *name;
    uint16_t                     flags;
    uint16_t                     function_id;
    uint8_t                      nb_parameters;
    uint8_t                      nb_definitions;
    int16_t                      deps;  /**< This is the index of the dependency array in the __DAGUE_object_t */
    dague_dependency_t           dependencies_goal;
    const symbol_t              *params[MAX_LOCAL_COUNT];
    const symbol_t              *locals[MAX_LOCAL_COUNT];
    const expr_t                *pred;
    const dague_flow_t          *in[MAX_PARAM_COUNT];
    const dague_flow_t          *out[MAX_PARAM_COUNT];
    const expr_t                *priority;
#if defined(DAGUE_SIM)
    dague_sim_cost_fct_t        *sim_cost_fct;
#endif
    dague_hook_t                *hook;
    dague_hook_t                *complete_execution;
    dague_traverse_function_t   *iterate_successors;
    dague_release_deps_t        *release_deps;
    dague_functionkey_fn_t      *key;
    char                        *body;
};

struct dague_data_pair_t {
    struct data_repo_entry   *data_repo;
    dague_arena_chunk_t      *data;
#if defined(HAVE_CUDA)
    struct _memory_elem      *mem2dev_data;
#endif  /* defined(HAVE_CUDA) */
};

/**
 * The minimal execution context contains only the smallest amount of information
 * required to be able to flow through the execution graph, by following data-flow
 * from one task to another. As an example, it contains the local variables but
 * not the data pairs. We need this in order to be able to only copy the minimal
 * amount of information when a new task is constructed.
 */
#define DAGUE_MINIMAL_EXECUTION_CONTEXT                  \
    dague_list_item_t        list_item;                  \
    struct dague_thread_mempool  *mempool_owner;         \
    dague_object_t          *dague_object;               \
    const  dague_function_t *function;                   \
    int32_t                  priority;                   \
    char *                   flowname;                   \
    assignment_t             locals[MAX_LOCAL_COUNT];

struct dague_minimal_execution_context_t {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
} dague_minimal_execution_context_t;

struct dague_execution_context_t {
    DAGUE_MINIMAL_EXECUTION_CONTEXT
#if defined(DAGUE_SIM)
    int                     sim_exec_date;
#endif
    dague_data_pair_t       data[MAX_PARAM_COUNT];
};

/**
 * Profiling data.
 */
#if defined(DAGUE_PROF_TRACE)
extern int schedule_poll_begin, schedule_poll_end;
extern int schedule_push_begin, schedule_push_end;
extern int schedule_sleep_begin, schedule_sleep_end;
extern int queue_add_begin, queue_add_end;
extern int queue_remove_begin, queue_remove_end;

#define DAGUE_PROF_FUNC_KEY_START(dague_object, function_index) \
    (dague_object)->profiling_array[2 * (function_index)]
#define DAGUE_PROF_FUNC_KEY_END(dague_object, function_index) \
    (dague_object)->profiling_array[1 + 2 * (function_index)]
#endif


/**
 * Dependencies management.
 */
typedef struct {
    int nb_released;
    uint32_t output_usage;
    struct data_repo_entry *output_entry;
    int action_mask;
    struct dague_remote_deps_t *deps;
    dague_execution_context_t** ready_lists;
#if defined(DISTRIBUTED)
    int remote_deps_count;
    struct dague_remote_deps_t *remote_deps;
#endif
} dague_release_dep_fct_arg_t;

dague_ontask_iterate_t dague_release_dep_fct(struct dague_execution_unit *eu,
                                             dague_execution_context_t *newcontext,
                                             dague_execution_context_t *oldcontext,
                                             int flow_index, int outdep_index,
                                             int rank_src, int rank_dst,
                                             int vpid_dst,
                                             dague_arena_t* arena,
                                             int nb_elt,
                                             void *param);

int dague_release_local_OUT_dependencies( dague_object_t *dague_object,
                                          dague_execution_unit_t* eu_context,
                                          const dague_execution_context_t* restrict origin,
                                          const dague_flow_t* restrict origin_flow,
                                          dague_execution_context_t* restrict exec_context,
                                          const dague_flow_t* restrict dest_flow,
                                          struct data_repo_entry* dest_repo_entry,
                                          dague_execution_context_t** pready_list );


/**
 * This is a convenience macro for the wrapper file. Do not call this destructor
 * directly from the applications, or face memory leaks as it only release the
 * most internal structues, while leaving the datatypes and the tasks management
 * buffers untouched. Instead, from the application layer call the _Destruct.
 */
#define DAGUE_INTERNAL_OBJECT_DESTRUCT(OBJ)             \
    do {                                                \
    dague_object_t* __obj = (dague_object_t*)(OBJ);     \
    __obj->object_destructor(__obj);                    \
    (OBJ) = NULL;                                       \
} while (0)

#define dague_execution_context_priority_comparator offsetof(dague_execution_context_t, priority)

#endif  /* DAGUE_INTERNAL_H_HAS_BEEN_INCLUDED */