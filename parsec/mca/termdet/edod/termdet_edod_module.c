/**
 * Copyright (c)      2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/include/parsec/execution_stream.h"
#include "parsec/utils/debug.h"
#include "parsec/mca/termdet/termdet.h"
#include "parsec/mca/termdet/edod/termdet_edod.h"
#include "parsec/remote_dep.h"

/**
 * Module functions
 */

static void parsec_termdet_edod_monitor_taskpool(parsec_taskpool_t *tp,
                                                 parsec_termdet_termination_detected_function_t cb);
static parsec_termdet_taskpool_state_t parsec_termdet_edod_taskpool_state(parsec_taskpool_t *tp);
static int parsec_termdet_edod_taskpool_ready(parsec_taskpool_t *tp);
static int parsec_termdet_edod_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_edod_taskpool_addto_nb_pa(parsec_taskpool_t *tp, int v);
static int parsec_termdet_edod_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_edod_taskpool_set_nb_pa(parsec_taskpool_t *tp, int v);

static int parsec_termdet_edod_outgoing_message_pack(parsec_taskpool_t *tp,
                                                     int dst_rank,
                                                     char *packed_buffer,
                                                     int *position,
                                                     int buffer_size,
                                                     MPI_Comm comm);
static int parsec_termdet_edod_outgoing_message_start(parsec_taskpool_t *tp,
                                                      int dst_rank,
                                                      parsec_remote_deps_t *remote_deps);
static int parsec_termdet_edod_incoming_message_start(parsec_taskpool_t *tp,
                                                      int src_rank,
                                                      char *packed_buffer,
                                                      int *position,
                                                      int buffer_size,
                                                      const parsec_remote_deps_t *msg,
                                                      MPI_Comm comm);
static int parsec_termdet_edod_incoming_message_end(parsec_taskpool_t *tp,
                                                    const parsec_remote_deps_t *msg);
static int parsec_termdet_edod_write_stats(parsec_taskpool_t *tp, FILE *fp);

const parsec_termdet_module_t parsec_termdet_edod_module = {
    &parsec_termdet_edod_component,
    {
        parsec_termdet_edod_monitor_taskpool,
        parsec_termdet_edod_taskpool_state,
        parsec_termdet_edod_taskpool_ready,
        parsec_termdet_edod_taskpool_addto_nb_tasks,
        parsec_termdet_edod_taskpool_addto_nb_pa,
        parsec_termdet_edod_taskpool_set_nb_tasks,
        parsec_termdet_edod_taskpool_set_nb_pa,
        0,
        parsec_termdet_edod_outgoing_message_start,
        parsec_termdet_edod_outgoing_message_pack,
        parsec_termdet_edod_incoming_message_start,
        parsec_termdet_edod_incoming_message_end,
        parsec_termdet_edod_write_stats
    }
};

/* In order to garbage collect when completing, and still differentiate between
 * terminated and not_monitored, we set the taskpool monitor to this constant after
 * detecting the termination. */
#define PARSEC_TERMDET_EDOD_TERMINATED ((void*)(0x1))

typedef struct parsec_termdet_edod_monitor_s {
    parsec_atomic_rwlock_t rw_lock;             /**< Operations that change the state take the write lock, operations that
                                                 *   read the state take the read lock */
    int      idle;                              /**< -1: not ready, 0/1: bool according to algorithm */
    int      free;
    int      inactive;
    int      child_inactive[2];
    uint32_t num_unack_msgs;

    uint32_t stats_nb_busy_idle;                /**< Statistics: number of transitions busy -> idle */
    uint32_t stats_nb_idle_busy;                /**< Statistics: number of transitions idle -> busy */
    uint32_t stats_nb_sent_msg;                 /**< Statistics: number of messages sent */
    uint32_t stats_nb_recv_msg;                 /**< Statistics: number of messages received */
    uint32_t stats_nb_sent_bytes;               /**< Statistics: number of bytes sent */
    uint32_t stats_nb_recv_bytes;               /**< Statistics: number of bytes received */
    struct timeval stats_time_start;
    struct timeval stats_time_last_idle;
    struct timeval stats_time_end;
} parsec_termdet_edod_monitor_t;

static void parsec_termdet_edod_termination_msg(const parsec_termdet_edod_empty_msg_t *msg, int src, parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm);
static void parsec_termdet_edod_stop_msg(const parsec_termdet_edod_empty_msg_t *msg, int src, parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm);
static void parsec_termdet_edod_resume_msg(const parsec_termdet_edod_tworanks_msg_t *msg, int src, parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm);
static void parsec_termdet_edod_acknowledge_msg(const parsec_termdet_edod_tworanks_msg_t *msg, int src, parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm);

int parsec_termdet_edod_msg_dispatch(int src, parsec_taskpool_t *tp, const void *msg, size_t size)
{
    parsec_termdet_edod_msg_type_t t = *(parsec_termdet_edod_msg_type_t*)msg;
    parsec_termdet_edod_monitor_t *tpm;
    const parsec_termdet_edod_tworanks_msg_t *tworanks_msg = (const parsec_termdet_edod_tworanks_msg_t*)msg;
    const parsec_termdet_edod_empty_msg_t *empty_msg = (const parsec_termdet_edod_empty_msg_t*)msg;
    (void)size;
    assert(NULL != tp);
    
    assert(NULL != tp->tdm.module);
    assert(&parsec_termdet_edod_module.module == tp->tdm.module);
    assert(PARSEC_TERMDET_EDOD_TERMINATED != tp->tdm.monitor);
    tpm = (parsec_termdet_edod_monitor_t *)tp->tdm.monitor;

    (void)size;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tReceived %d bytes from %d relative to taskpool %d",
                         size, src, tp->taskpool_id);
    if( src != tp->context->my_rank ) {
        tpm->stats_nb_recv_msg++;
        tpm->stats_nb_recv_bytes+=sizeof(int)+size;
    }

    switch( t ) {
    case PARSEC_TERMDET_EDOD_TERMINATION_MSG:
        assert( size == sizeof(parsec_termdet_edod_empty_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tIt is a TERMINATION message");
        parsec_termdet_edod_termination_msg( empty_msg, src, tp, tpm );
        return PARSEC_SUCCESS;
    case PARSEC_TERMDET_EDOD_STOP_MSG:
        assert( size == sizeof(parsec_termdet_edod_empty_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tIt is a STOP message");
        parsec_termdet_edod_stop_msg( empty_msg, src, tp, tpm );
        return PARSEC_SUCCESS;
    case PARSEC_TERMDET_EDOD_RESUME_MSG:
        assert( size == sizeof(parsec_termdet_edod_tworanks_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tIt is a RESUME message");
        parsec_termdet_edod_resume_msg( tworanks_msg, src, tp, tpm );
        return PARSEC_SUCCESS;
    case PARSEC_TERMDET_EDOD_ACKNOWLEDGE_MSG:
        assert( size == sizeof(parsec_termdet_edod_tworanks_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tIt is a ACKNOWLEDGE message");
        parsec_termdet_edod_acknowledge_msg( tworanks_msg, src, tp, tpm );
        return PARSEC_SUCCESS;
    }
    assert(0);
    return PARSEC_ERROR;
}

static int parsec_termdet_edod_topology_nb_children(parsec_taskpool_t *tp)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;

    if( 2*context->my_rank + 2 < context->nb_nodes )
        return 2;
    if( 2*context->my_rank + 1 < context->nb_nodes )
        return 1;
    return 0;
}

static int parsec_termdet_edod_topology_is_root(parsec_taskpool_t *tp)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;
    return context->my_rank == 0;
}

static int parsec_termdet_edod_topology_child(parsec_taskpool_t *tp, int i)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;

    assert(i == 0 || i == 1);
    assert( 2*context->my_rank + i + 1 < context->nb_nodes);
    return 2 * context->my_rank + i + 1;
}

static int parsec_termdet_edod_topology_parent(parsec_taskpool_t *tp)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;

    assert(context->my_rank > 0);
    return (context->my_rank-1) >> 1;
}

static int parsec_termdet_edod_topology_child_path(parsec_taskpool_t *tp, int dst)
{
    int l, r;
    int d, p;
    assert(dst > 0);
    assert( parsec_termdet_edod_topology_nb_children(tp) > 0 );
    assert( parsec_termdet_edod_topology_nb_children(tp) <= 2 );

    d = dst;
    l = parsec_termdet_edod_topology_child(tp, 0);
    if(d == l)
        return l;
    if( parsec_termdet_edod_topology_nb_children(tp) == 2 ) {
        r =  parsec_termdet_edod_topology_child(tp, 1);
        if(r == d)
            return r;
    } else {
        r = l;
    }
    
    p = (d - 1)/2;
    while(d > tp->context->my_rank) {
        if(p == l)
            return l;
        if(p == r)
            return r;
        d = p;
        p = (d-1)/2;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                         "TERMDET-EDOD:\t Internal error, impossible to find path from %d to %d using children path",
                         tp->context->my_rank, dst);
    assert(0);
    return -1;
}

static void parsec_termdet_edod_monitor_taskpool(parsec_taskpool_t *tp,
                                                        parsec_termdet_termination_detected_function_t cb)
{
    parsec_termdet_edod_monitor_t *tpm;
    int i;
    assert(&parsec_termdet_edod_module.module == tp->tdm.module);
    tpm = (parsec_termdet_edod_monitor_t*)malloc(sizeof(parsec_termdet_edod_monitor_t));
    tp->tdm.callback = cb;
    tpm->idle = -1;
    tpm->free = 0;
    tpm->inactive = 0;
    for(i = 0; i < 2; i++)
        tpm->child_inactive[i] = 0;
    tpm->num_unack_msgs = 0;

    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tProcess initializes state to NOT_READY");
    tp->tdm.monitor = tpm;

    tpm->stats_nb_busy_idle = 0;
    tpm->stats_nb_idle_busy = 0;
    tpm->stats_nb_sent_msg = 0;
    tpm->stats_nb_recv_msg = 0;
    tpm->stats_nb_sent_bytes = 0;
    tpm->stats_nb_recv_bytes = 0;

    tp->tdm.counters.nb_tasks = 0;
    tp->tdm.counters.nb_pending_actions = 0;

    parsec_atomic_rwlock_init(&tpm->rw_lock);

    gettimeofday(&tpm->stats_time_start, NULL);
}

static parsec_termdet_taskpool_state_t parsec_termdet_edod_taskpool_state(parsec_taskpool_t *tp)
{
    parsec_termdet_edod_monitor_t *tpm;
    parsec_termdet_taskpool_state_t state;
    if( tp->tdm.module == NULL )
        return PARSEC_TERM_TP_NOT_MONITORED;
    assert(tp->tdm.module == &parsec_termdet_edod_module.module);
    if( tp->tdm.monitor == PARSEC_TERMDET_EDOD_TERMINATED )
        return PARSEC_TERM_TP_TERMINATED;
    tpm = tp->tdm.monitor;
    parsec_atomic_rwlock_rdlock(&tpm->rw_lock);
    if( tpm->idle == -1 ) {
        state = PARSEC_TERM_TP_NOT_READY;
        assert( parsec_termdet_edod_topology_nb_children(tp) <= 2 );
    } else {
        if( tpm->idle ) {
            state = PARSEC_TERM_TP_IDLE;
        } else {
            state = PARSEC_TERM_TP_BUSY;
        }
    }
    parsec_atomic_rwlock_rdunlock(&tpm->rw_lock);
    return state;
}

static int parsec_termdet_edod_taskpool_ready(parsec_taskpool_t *tp)
{
    parsec_termdet_edod_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );
    tpm = (parsec_termdet_edod_monitor_t*)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    assert( tpm->idle == -1);
    tpm->idle = 0;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tProcess changed state for BUSY (taskpool ready)");
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    return PARSEC_SUCCESS;
}

static void signal_termination(parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm)
{
    int i;
    for(i = 0; i < parsec_termdet_edod_topology_nb_children(tp); i++) {
        parsec_termdet_edod_empty_msg_t term_msg;
        term_msg.msg_type = PARSEC_TERMDET_EDOD_TERMINATION_MSG;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tSending TERMINATION message to rank %d",
                             parsec_termdet_edod_topology_child(tp, i));
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_edod_empty_msg_t) + sizeof(int);
        parsec_comm_send_message(parsec_termdet_edod_topology_child(tp, i),
                                 parsec_termdet_edod_msg_tag,
                                 tp,
                                 &term_msg, sizeof(parsec_termdet_edod_empty_msg_t));
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tTERMINATION detected");
    parsec_termdet_edod_write_stats(tp, stdout);
    tp->tdm.monitor = PARSEC_TERMDET_EDOD_TERMINATED;
    tp->tdm.callback(tp);
    free(tpm);
}

static void parsec_termdet_edod_check_state_state_changed(parsec_termdet_edod_monitor_t *tpm, parsec_taskpool_t *tp)
{
    int all_child = 1;
    int i;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tEntering check state with idle = %d, free = %d, num_unack_msgs = %d, child_inactive[0] = %d, child_inactive[1] = %d, inactive = %d",
                         tpm->idle, tpm->free, tpm->num_unack_msgs, tpm->child_inactive[0], tpm->child_inactive[1], tpm->inactive);
    if( tpm->idle == 1 && tpm->num_unack_msgs == 0 ) {
        tpm->free = 1;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tIn check state, free set to 1");
    }
    for(i = 0; i < parsec_termdet_edod_topology_nb_children(tp); i++)
        if( tpm->child_inactive[i] == 0 ) {
            all_child = 0;
            break;
        }
    if( all_child && tpm->free && (tpm->inactive == 0) ) {
        if( parsec_termdet_edod_topology_is_root(tp) ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tIn check state, signaling termination");
            gettimeofday(&tpm->stats_time_end, NULL);
            signal_termination(tp, tpm);
        } else {
            parsec_termdet_edod_tworanks_msg_t stop_msg;
            stop_msg.msg_type = PARSEC_TERMDET_EDOD_STOP_MSG;
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tSending STOP message to rank %d",
                                 parsec_termdet_edod_topology_parent(tp));
            tpm->stats_nb_sent_msg++;
            tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_edod_tworanks_msg_t) + sizeof(int);
            parsec_comm_send_message(parsec_termdet_edod_topology_parent(tp),
                                     parsec_termdet_edod_msg_tag,
                                     tp,
                                     &stop_msg, sizeof(parsec_termdet_edod_empty_msg_t));
        }
        tpm->inactive = 1;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tIn check state, inactive set to 1");
    } else {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tLeaving check state");
    }
}

static void parsec_termdet_edod_check_state_workload_changed(parsec_termdet_edod_monitor_t *tpm, parsec_taskpool_t *tp)
{
    if( tp->tdm.counters.atomic == 0 ) {
        tpm->idle = 1;
        tpm->stats_nb_busy_idle++;
        gettimeofday(&tpm->stats_time_last_idle, NULL);
        parsec_termdet_edod_check_state_state_changed(tpm, tp);
    }
}

static int parsec_termdet_edod_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_edod_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );
    assert( v >= 0 );
    tpm = (parsec_termdet_edod_monitor_t *)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    if( (int)tp->tdm.counters.nb_tasks != v) {
        tp->tdm.counters.nb_tasks = v;
        parsec_termdet_edod_check_state_workload_changed(tpm, tp);
    }
    if( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    return v;
}

static int parsec_termdet_edod_taskpool_set_nb_pa(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_edod_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );
    assert( v >= 0 );
    tpm = (parsec_termdet_edod_monitor_t *)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    if( (int)tp->tdm.counters.nb_pending_actions != v) {
        tp->tdm.counters.nb_pending_actions = v;
        parsec_termdet_edod_check_state_workload_changed(tpm, tp);
    }
    if( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    return v;
}

static int parsec_termdet_edod_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_edod_monitor_t *tpm;
    int ret;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );
    if(v == 0)
        return tp->tdm.counters.nb_tasks;
    tpm = (parsec_termdet_edod_monitor_t *)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    tp->tdm.counters.nb_tasks += v;
    ret = tp->tdm.counters.nb_tasks;
    parsec_termdet_edod_check_state_workload_changed(tpm, tp);
    if( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    return ret;
}

static int parsec_termdet_edod_taskpool_addto_nb_pa(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_edod_monitor_t *tpm;
    int ret;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );
    if(v == 0)
        return tp->tdm.counters.nb_pending_actions;
    tpm = (parsec_termdet_edod_monitor_t *)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    tp->tdm.counters.nb_pending_actions += v;
    ret = tp->tdm.counters.nb_pending_actions;
    parsec_termdet_edod_check_state_workload_changed(tpm, tp);
    if( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    return ret;
}

static int parsec_termdet_edod_outgoing_message_start(parsec_taskpool_t *tp,
                                                      int dst_rank,
                                                      parsec_remote_deps_t *remote_deps)
{
    parsec_termdet_edod_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );
    (void)dst_rank;
    (void)remote_deps;
    tpm = tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    tpm->num_unack_msgs++;
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);

    return 1;
}

static int parsec_termdet_edod_outgoing_message_pack(parsec_taskpool_t *tp,
                                                            int dst_rank,
                                                            char *packed_buffer,
                                                            int *position,
                                                            int buffer_size,
                                                            MPI_Comm comm)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );
    /* No piggybacking */
    (void)tp;
    (void)dst_rank;
    (void)packed_buffer;
    (void)position;
    (void)buffer_size;
    (void)comm;
    return PARSEC_SUCCESS;
}

static int parsec_termdet_edod_incoming_message_start(parsec_taskpool_t *tp,
                                                      int src_rank,
                                                      char *packed_buffer,
                                                      int *position,
                                                      int buffer_size,
                                                      const parsec_remote_deps_t *msg,
                                                      MPI_Comm comm)
{
    parsec_termdet_edod_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );
    
    tpm = tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    tpm->idle = 0;
    tpm->stats_nb_idle_busy++;
    tpm->free = 0;
    if( tpm->inactive == 1 ) {
        parsec_termdet_edod_tworanks_msg_t resume_msg;
        resume_msg.msg_type = PARSEC_TERMDET_EDOD_RESUME_MSG;
        resume_msg.first  = src_rank;
        resume_msg.second = tp->context->my_rank;
        tpm->inactive = 0;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tSending RESUME(%d, %d) message to rank %d",
                             resume_msg.first, resume_msg.second, parsec_termdet_edod_topology_parent(tp));
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_edod_tworanks_msg_t) + sizeof(int);
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
        parsec_comm_send_message(parsec_termdet_edod_topology_parent(tp),
                                 parsec_termdet_edod_msg_tag,
                                 tp,
                                 &resume_msg, sizeof(parsec_termdet_edod_tworanks_msg_t));
    } else {
        parsec_termdet_edod_tworanks_msg_t ack_msg;
        ack_msg.msg_type = PARSEC_TERMDET_EDOD_ACKNOWLEDGE_MSG;
        ack_msg.first  = src_rank;
        ack_msg.second = tp->context->my_rank;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tSending ACKNOWLEDGE(%d, %d) message to rank %d",
                             ack_msg.first, ack_msg.second, src_rank);
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_edod_tworanks_msg_t) + sizeof(int);
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
        parsec_comm_send_message(src_rank,
                                 parsec_termdet_edod_msg_tag,
                                 tp,
                                 &ack_msg, sizeof(parsec_termdet_edod_tworanks_msg_t));
    }

    /* No piggybacking */
    (void)src_rank;
    (void)packed_buffer;
    (void)position;
    (void)buffer_size;
    (void)msg;
    (void)comm;
    
    return PARSEC_SUCCESS;
}

static int parsec_termdet_edod_incoming_message_end(parsec_taskpool_t *tp,
                                                    const parsec_remote_deps_t *msg)
{
    (void)msg;

    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_edod_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED );

    (void)tp;
    (void)msg;
    
    return PARSEC_SUCCESS;
}

static int parsec_termdet_edod_write_stats(parsec_taskpool_t *tp, FILE *fp)
{
    parsec_termdet_edod_monitor_t *tpm;
    struct timeval t1, t2;
    assert(NULL != tp->tdm.module);
    assert(&parsec_termdet_edod_module.module == tp->tdm.module);
    tpm = (parsec_termdet_edod_monitor_t *)tp->tdm.monitor;

    timersub(&tpm->stats_time_end, &tpm->stats_time_last_idle, &t1);
    timersub(&tpm->stats_time_end, &tpm->stats_time_start, &t2);
    
    fprintf(fp, "NP: %d M: EDOD Rank: %d Taskpool#: %d #Transitions_Busy_to_Idle: %u #Transitions_Idle_to_Busy: %u #Times_Credit_was_Borrowed: 0 #Times_Credit_was_Flushed: 0 #Times_a_message_was_Delayed: 0 #Times_credit_was_merged: 0 #SentCtlMsg: %u #RecvCtlMsg: %u SentCtlBytes: %u RecvCtlBytes: %u WallTime: %u.%06u Idle2End: %u.%06u\n",
            tp->context->nb_nodes,
            tp->context->my_rank,
            tp->taskpool_id,
            tpm->stats_nb_busy_idle,
            tpm->stats_nb_idle_busy,
            tpm->stats_nb_sent_msg,
            tpm->stats_nb_recv_msg,
            tpm->stats_nb_sent_bytes,
            tpm->stats_nb_recv_bytes,
            (unsigned int)t2.tv_sec, (unsigned int)t2.tv_usec,
            (unsigned int)t1.tv_sec, (unsigned int)t1.tv_usec);

    return PARSEC_SUCCESS;
}

static void parsec_termdet_edod_termination_msg(const parsec_termdet_edod_empty_msg_t *msg, int src, parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm)
{
    (void)src;
    (void)msg;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    gettimeofday(&tpm->stats_time_end, NULL);
    signal_termination(tp, tpm);
    assert(tp->tdm.monitor == PARSEC_TERMDET_EDOD_TERMINATED);
}

static void parsec_termdet_edod_stop_msg(const parsec_termdet_edod_empty_msg_t *msg, int src, parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm)
{
    int l;
    (void)msg;
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    for(l = 0; l < parsec_termdet_edod_topology_nb_children(tp); l++) {
        if( src == parsec_termdet_edod_topology_child(tp, l) ) {
            tpm->child_inactive[l]++;
            parsec_termdet_edod_check_state_state_changed(tpm, tp);
            if(tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED)
                parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
            return;
        }
    }
    assert(0);
}

static void parsec_termdet_edod_resume_msg(const parsec_termdet_edod_tworanks_msg_t *msg, int src, parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm)
{
    int l;

    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    for(l = 0; l < parsec_termdet_edod_topology_nb_children(tp); l++) {
        if( src == parsec_termdet_edod_topology_child(tp, l) ) {
            tpm->child_inactive[l]--;
            parsec_termdet_edod_check_state_state_changed(tpm, tp);
            if(tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED)
                parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
            if( tpm->inactive ) {
                parsec_termdet_edod_tworanks_msg_t resume_msg;
                resume_msg.msg_type = PARSEC_TERMDET_EDOD_RESUME_MSG;
                resume_msg.first  = msg->first;
                resume_msg.second = msg->second;
                PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tSending RESUME(%d, %d) message to rank %d",
                                     resume_msg.first, resume_msg.second, parsec_termdet_edod_topology_parent(tp));
                tpm->stats_nb_sent_msg++;
                tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_edod_tworanks_msg_t) + sizeof(int);
                parsec_comm_send_message(parsec_termdet_edod_topology_parent(tp),
                                         parsec_termdet_edod_msg_tag,
                                         tp,
                                         &resume_msg, sizeof(parsec_termdet_edod_tworanks_msg_t));
                tpm->inactive = 0;
            } else {
                parsec_termdet_edod_tworanks_msg_t ack_msg;
                ack_msg.msg_type = PARSEC_TERMDET_EDOD_ACKNOWLEDGE_MSG;
                ack_msg.first  = msg->first;
                ack_msg.second = msg->second;
                PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tSending ACKNOWLEDGE(%d, %d) message to rank %d",
                                     ack_msg.first, ack_msg.second, parsec_termdet_edod_topology_child_path(tp, ack_msg.second));
                tpm->stats_nb_sent_msg++;
                tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_edod_tworanks_msg_t) + sizeof(int);
                parsec_comm_send_message(parsec_termdet_edod_topology_child_path(tp, ack_msg.second),
                                         parsec_termdet_edod_msg_tag,
                                         tp,
                                         &ack_msg, sizeof(parsec_termdet_edod_tworanks_msg_t));
            }
            return;
        }
    }
    assert(0);
}

static void parsec_termdet_edod_acknowledge_msg(const parsec_termdet_edod_tworanks_msg_t *msg, int src, parsec_taskpool_t *tp, parsec_termdet_edod_monitor_t *tpm)
{
    (void)src;

    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    if( msg->first == tp->context->my_rank ) {
        tpm->num_unack_msgs--;
        parsec_termdet_edod_check_state_state_changed(tpm, tp);
        if(tp->tdm.monitor != PARSEC_TERMDET_EDOD_TERMINATED)
            parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    } else if(msg->second == tp->context->my_rank) {
        parsec_termdet_edod_tworanks_msg_t ack_msg;
        ack_msg.msg_type = PARSEC_TERMDET_EDOD_ACKNOWLEDGE_MSG;
        ack_msg.first  = msg->first;
        ack_msg.second = msg->second;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tSending ACKNOWLEDGE(%d, %d) message to rank %d",
                             ack_msg.first, ack_msg.second, msg->first);
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_edod_tworanks_msg_t) + sizeof(int);
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);        
        parsec_comm_send_message(msg->first,
                                 parsec_termdet_edod_msg_tag,
                                 tp,
                                 &ack_msg, sizeof(parsec_termdet_edod_tworanks_msg_t));
    } else {
        parsec_termdet_edod_tworanks_msg_t ack_msg;
        ack_msg.msg_type = PARSEC_TERMDET_EDOD_ACKNOWLEDGE_MSG;
        ack_msg.first  = msg->first;
        ack_msg.second = msg->second;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-EDOD:\tSending ACKNOWLEDGE(%d, %d) message to rank %d",
                             ack_msg.first, ack_msg.second, parsec_termdet_edod_topology_child_path(tp, ack_msg.second));
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_edod_tworanks_msg_t) + sizeof(int);
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);        
        parsec_comm_send_message(parsec_termdet_edod_topology_child_path(tp, ack_msg.second),
                                 parsec_termdet_edod_msg_tag,
                                 tp,
                                 &ack_msg, sizeof(parsec_termdet_edod_tworanks_msg_t));
    }
}
