/*
 * Copyright (c) 2009-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_H_HAS_BEEN_INCLUDED
#define DAGUE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

BEGIN_C_DECLS

typedef struct dague_handle_s            dague_handle_t;
typedef struct dague_execution_context_s dague_execution_context_t;
typedef struct dague_execution_unit_s    dague_execution_unit_t;
/**< The general context that holds all the threads of dague for this MPI process */
typedef struct dague_context_s           dague_context_t;
typedef struct dague_arena_s             dague_arena_t;

/**
 * TO BE REMOVED.
 */
typedef void* (*dague_data_allocate_t)(size_t matrix_size);
typedef void (*dague_data_free_t)(void *data);
extern dague_data_allocate_t dague_data_allocate;
extern dague_data_free_t     dague_data_free;

/**
 * CONTEXT MANIPULATION FUNCTIONS.
 */

/**
 * Create a new execution context, using the number of resources passed
 * with the arguments. Every execution happend in the context of such an
 * execution context. Several contextes can cohexist on disjoint resources
 * in same time.
 */
dague_context_t* dague_init( int nb_cores, int* pargc, char** pargv[]);

/**
 * Reset the remote_dep comm engine associated with @context, and use
 * the communication context @opaque_comm_ctx in the future
 * (typically an MPI communicator);
 *   dague_context_wait becomes collective accross nodes spanning on this
 *   communication context.
 */
int dague_remote_dep_set_ctx( dague_context_t* context, void* opaque_comm_ctx );


/**
 * Complete all pending operations on the execution context, and release
 * all associated resources. Threads and acclerators attached to this
 * context will be released.
 */
int dague_fini( dague_context_t** pcontext );

/**
 * Attach an execution handle on a context, in other words on the set of
 * resources associated to this particular context. A matching between
 * the capabilitis of the context and the support from the handle will be
 * done during this step, which will basically define if accelerators can
 * be used for the execution.
 *
 * @param [INOUT] The dague context where the tasks generated by the dague_handle_t
 *                are to be executed.
 * @param [INOUT] The dague object with pending tasks.
 *
 * @return 0 If the enqueue operation succeeded.
 */
int dague_enqueue( dague_context_t* , dague_handle_t* );

/**
 * Start the runtime by allowing all the other threads to start executing.
 * This call should be paired with one of the completion calls, test or wait.
 *
 * @returns: 0 if the other threads in this context have been started, -1 if the
 * context was already active, -2 if there was nothing to do and no threads hav
 * been activated.
 */
int dague_context_start(dague_context_t* context);

/**
 * Check the status of an ongoing execution, started with dague_start
 *
 * @param [INOUT] The dague context where the execution is taking place.
 *
 * @return 0 If the execution is still ongoing.
 * @return 1 If the execution is completed, and the dague_context has no
 *           more pending tasks. All subsequent calls on the same context
 *           will automatically succeed.
 */
int dague_context_test( dague_context_t* context );

/**
 * Progress the execution context until no further operations are available.
 * Upon return from this function, all resources (threads and accelerators)
 * associated with the corresponding context are put in a mode where they are
 * not active. New handles enqueued during the progress stage are automatically
 * taken into account, and the caller of this function will not return to the
 * user until all pending handles are completed and all other threads are in a
 * sleeping mode.
 *
 * @param [INOUT] The dague context where the execution is taking place.
 *
 * @return 0 If the execution is completed.
 * @return * Any other error raised by the tasks themselves.
 */
int dague_context_wait(dague_context_t* context);

/**
 * HANDLE MANIPULATION FUNCTIONS.
 */

/**
 * The completion callback of a dague_handle. Once the handle has been
 * completed, i.e. all the local tasks associated with the handle have
 * been executed, and before the handle is marked as done, this callback
 * will be triggered. Inside the callback the handle should not be
 * modified.
 */
typedef int (*dague_event_cb_t)(dague_handle_t* dague_handle, void*);

/* Accessors to set and get the completion callback and data */
int dague_set_complete_callback(dague_handle_t* dague_handle,
                                dague_event_cb_t complete_cb, void* complete_data);
int dague_get_complete_callback(const dague_handle_t* dague_handle,
                                dague_event_cb_t* complete_cb, void** complete_data);
/* Accessors to set and get the enqueue callback and data */
int dague_set_enqueue_callback(dague_handle_t* dague_handle,
                               dague_event_cb_t enqueue_cb, void* enqueue_data);
int dague_get_enqueue_callback(const dague_handle_t* dague_handle,
                               dague_event_cb_t* enqueue_cb, void** enqueue_data);

/**< Retrieve the local object attached to a unique object id */
dague_handle_t* dague_handle_lookup(uint32_t handle_id);
/**< Reverse an unique ID for the handle. Beware that on a distributed environment the
 * connected objects must have the same ID.
 */
int dague_handle_reserve_id(dague_handle_t* handle);
/**< Register the object with the engine. The object must have a unique handle, especially
 * in a distributed environment.
 */
int dague_handle_register(dague_handle_t* handle);
/**< Unregister the object with the engine. This make the handle available for
 * future handles. Beware that in a distributed environment the connected objects
 * must have the same ID.
 */
void dague_handle_unregister(dague_handle_t* handle);
/**< globally synchronize object id's so that next register generates the same
 *  id at all ranks. */
void dague_handle_sync_ids(void);

/**
 * Compose sequentially two handles. If start is already a composed
 * object, then next will be added sequentially to the list. These
 * handles will execute one after another as if there were sequential.
 * The resulting compound dague_handle is returned.
 */
dague_handle_t* dague_compose(dague_handle_t* start, dague_handle_t* next);

/**< Free the resource allocated in the dague handle. The handle should be unregistered first. */
void dague_handle_free(dague_handle_t *handle);

/**<
 * The final step of a handle activation. At this point we assume that all the local
 * initializations have been succesfully completed for all components, and that the
 * handle is ready to be registered in the system, and any potential pending tasks
 * ready to go.
 *
 * The local_task allows for concurrent management of the startup_queue, and provide a way
 * to prevent a task from being added to the scheduler. The execution unit eu, is only
 * meaningful if there are any tasks to be scheduled. The nb_tasks is used to detect
 * if the handle should be registered with the communication engine or not.
 */
int dague_handle_enable(dague_handle_t* handle,
                        dague_execution_context_t** startup_queue,
                        dague_execution_context_t* local_task,
                        dague_execution_unit_t * eu,
                        int nb_tasks);

/**< Update the number of tasks by adding the increment (if the increment is negative
 * the number of tasks is decreased).
 */
void dague_handle_update_nbtask( dague_handle_t* handle, int32_t nb_tasks );

/**< Print DAGuE usage message */
void dague_usage(void);

/**
 * Allow to change the default priority of an object. It returns the
 * old priority (the default priorityy of an object is 0). This function
 * can be used during the lifetime of an object, however, only tasks
 * generated after this call will be impacted.
 */
int32_t dague_set_priority( dague_handle_t* object, int32_t new_priority );

/* Dump functions */
char* dague_snprintf_execution_context( char* str, size_t size,
                                        const dague_execution_context_t* task);
struct dague_function_s;
struct assignment_s;
char* dague_snprintf_assignments( char* str, size_t size,
                                  const struct dague_function_s* function,
                                  const struct assignment_s* locals);

END_C_DECLS

#endif  /* DAGUE_H_HAS_BEEN_INCLUDED */
