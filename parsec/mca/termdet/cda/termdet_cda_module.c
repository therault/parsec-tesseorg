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
#include "parsec/mca/termdet/cda/termdet_cda.h"
#include "parsec/remote_dep.h"
#include "parsec/class/dequeue.h"

#undef assert
#ifndef _NDEBUG
#define assert(__t) do {                                \
        if(!(__t)) {                                    \
            volatile int loop = 1;                      \
            char hostname[256];                         \
            gethostname(hostname, 256);                 \
            fprintf(stderr,                             \
                    "Assertion '%s' at %s:%d failed\n"  \
                    "ssh -t %s gdb -p %d\n",            \
                    #__t,  __FILE__, __LINE__,          \
                    hostname, getpid() );               \
            while(loop) {                               \
                sleep(1);                               \
            }                                           \
        }                                               \
    } while(0)
#else
#define assert(__) do {} while(0)
#endif

/**
 * Module functions
 */

static void parsec_termdet_cda_monitor_taskpool(parsec_taskpool_t *tp,
                                                parsec_termdet_termination_detected_function_t cb);
static parsec_termdet_taskpool_state_t parsec_termdet_cda_taskpool_state(parsec_taskpool_t *tp);
static int parsec_termdet_cda_taskpool_ready(parsec_taskpool_t *tp);
static int parsec_termdet_cda_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_cda_taskpool_addto_nb_pa(parsec_taskpool_t *tp, int v);
static int parsec_termdet_cda_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_cda_taskpool_set_nb_pa(parsec_taskpool_t *tp, int v);

static int parsec_termdet_cda_outgoing_message_pack(parsec_taskpool_t *tp,
                                                    int dst_rank,
                                                    char *packed_buffer,
                                                    int *position,
                                                    int buffer_size,
                                                    MPI_Comm comm);
static int parsec_termdet_cda_outgoing_message_start(parsec_taskpool_t *tp,
                                                     int dst_rank,
                                                     parsec_remote_deps_t *remote_deps);
static int parsec_termdet_cda_incoming_message_start(parsec_taskpool_t *tp,
                                                     int src_rank,
                                                     char *packed_buffer,
                                                     int *position,
                                                     int buffer_size,
                                                     const parsec_remote_deps_t *msg,
                                                     MPI_Comm comm);
static int parsec_termdet_cda_incoming_message_end(parsec_taskpool_t *tp,
                                                   const parsec_remote_deps_t *msg);
static int parsec_termdet_cda_write_stats(parsec_taskpool_t *tp, FILE *fp);

const parsec_termdet_module_t parsec_termdet_cda_module = {
    &parsec_termdet_cda_component,
    {
        parsec_termdet_cda_monitor_taskpool,
        parsec_termdet_cda_taskpool_state,
        parsec_termdet_cda_taskpool_ready,
        parsec_termdet_cda_taskpool_addto_nb_tasks,
        parsec_termdet_cda_taskpool_addto_nb_pa,
        parsec_termdet_cda_taskpool_set_nb_tasks,
        parsec_termdet_cda_taskpool_set_nb_pa,
        sizeof(uint64_t),
        parsec_termdet_cda_outgoing_message_start,
        parsec_termdet_cda_outgoing_message_pack,
        parsec_termdet_cda_incoming_message_start,
        parsec_termdet_cda_incoming_message_end,
        parsec_termdet_cda_write_stats
    }
};

/* In order to garbage collect when completing, and still differentiate between
 * terminated and not_monitored, we set the taskpool monitor to this constant after
 * detecting the termination. */
#define PARSEC_TERMDET_CDA_TERMINATED ((void*)(0x1))

#define PARSEC_TERMDET_CDA_INITIAL_CREDIT_PER_RANK 0xFFFFFFFFl
#define PARSEC_TERMDET_CDA_NOT_READY               0xFFFFFFFFFFFFFFl

typedef struct {
    parsec_list_item_t super;
    int dst_rank;
    parsec_remote_deps_t *deps;
} parsec_termdet_cda_delayed_msg_t;

PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_termdet_cda_delayed_msg_t);
OBJ_CLASS_INSTANCE(parsec_termdet_cda_delayed_msg_t, parsec_list_item_t, NULL, NULL);


typedef struct parsec_termdet_cda_monitor_s {
    parsec_atomic_rwlock_t lock;                /**< Atomic lock to update status atomically
                                                 *   Lock is taken read for state computation,
                                                 *   write if any variable changes */
    int root;                                   /**< What rank is the root */
    uint32_t committed;                         /**< How many parts of the credit are currently committed */
    uint64_t credit;                            /**< Current available credit */
    union {
        uint64_t root_missing_credit;           /**< For root, how many credit is still in the system -- When this reaches 0, termination is detected. */
        uint64_t other_borrowing;               /**< For other processes, are we borrowing at this time? Used to avoid multiple borrow requests in parallel (0/1) */
        uint64_t not_ready;                     /**< Before we are ready, this has the value PARSEC_TERMDET_CDA_NOT_READY (both for root and others) */
    };
    parsec_dequeue_t delayed_messages;          /**< Dequeue of messages that we cannot send because we are missing credit */
    uint32_t flush_id;                          /**< Flush identifier. When a process decides to flush, we want to
                                                 *   be sure that no incoming message in the MPI queues is an activation message.
                                                 *   To ensure that, we create a flush_msg for ourselves, carrying this flush_id.
                                                 *   When we become active, we increment the flush_id.
                                                 *   If we receive the message with the same flush_id, it means that we did a full
                                                 *   cycle of MPI message processing without being activated, so we flush.
                                                 *   Otherwise, we simply discard the flush */
    
    uint32_t stats_nb_borrowed;                 /**< Statistics: number of times some credit had to be borrowed */
    uint32_t stats_nb_flush;                    /**< Statistics: number of times the credit was flushed to the root */
    uint32_t stats_nb_delayed;                  /**< Statistics: number of times a message was delayed */
    uint32_t stats_nb_credit_merge;             /**< Statistics: number of times credit returning from a child was merged */
    uint32_t stats_nb_busy_idle;                /**< Statistics: number of transitions busy -> idle */
    uint32_t stats_nb_idle_busy;                /**< Statistics: number of transitions idle -> busy */
    uint32_t stats_nb_sent_msg;                 /**< Statistics: number of messages sent */
    uint32_t stats_nb_recv_msg;                 /**< Statistics: number of messages received */
    uint32_t stats_nb_sent_bytes;               /**< Statistics: number of bytes sent */
    uint32_t stats_nb_recv_bytes;               /**< Statistics: number of bytes received */
    struct timeval stats_time_start;
    struct timeval stats_time_last_idle;
    struct timeval stats_time_end;
} parsec_termdet_cda_monitor_t;

static uint64_t parsec_termdet_cda_split_credit(uint64_t credit, uint32_t committed)
{
    assert(committed >= 1);
    assert(credit >= committed);
#undef TERMDET_CDA_USE_DIVIDE
#ifdef TERMDET_CDA_USE_DIVIDE
    return credit / committed;
#else
    if( committed == 1 )
        return credit;
    if( credit - credit/committed > 4096 )
        return credit / committed;
    if( (credit-64) / (committed-1) > 0 ) {
        return 64;
    }
    return credit/committed;
#endif
}

static int parsec_termdet_cda_topology_nb_children(parsec_taskpool_t *tp)
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

#if 0
static int parsec_termdet_cda_topology_is_root(parsec_taskpool_t *tp)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;
    return context->my_rank == 0;
}
#endif

static int parsec_termdet_cda_topology_child(parsec_taskpool_t *tp, int i)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;

    assert(i == 0 || i == 1);
    assert( 2*context->my_rank + i + 1 < context->nb_nodes);
    return 2 * context->my_rank + i + 1;
}

static int parsec_termdet_cda_topology_parent(parsec_taskpool_t *tp)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;

    assert(context->my_rank > 0);
    return (context->my_rank-1) >> 5;
}

static void parsec_termdet_cda_termination_msg(int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm);
static void parsec_termdet_cda_credit_back_msg(const parsec_termdet_cda_credit_carrying_msg_t *credit_msg, int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm);
static void parsec_termdet_cda_borrow_credit_msg(int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm);
static void parsec_termdet_cda_give_credits_to_other_msg(const parsec_termdet_cda_credit_carrying_msg_t *credit_msg, int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm);
static void parsec_termdet_cda_flush_msg(const parsec_termdet_cda_flush_msg_t *flush_msg, int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm);

int parsec_termdet_cda_msg_dispatch(int src, parsec_taskpool_t *tp, const void *msg, size_t size)
{
    parsec_termdet_cda_msg_type_t t = *(parsec_termdet_cda_msg_type_t*)msg;
    const parsec_termdet_cda_credit_carrying_msg_t *credit_msg = (const parsec_termdet_cda_credit_carrying_msg_t*)msg;
    const parsec_termdet_cda_flush_msg_t *flush_msg = (const parsec_termdet_cda_flush_msg_t*)msg;
    parsec_termdet_cda_monitor_t *tpm;

    assert(NULL != tp->tdm.module);
    assert(&parsec_termdet_cda_module.module == tp->tdm.module);
    assert(PARSEC_TERMDET_CDA_TERMINATED != tp->tdm.monitor);
    tpm = (parsec_termdet_cda_monitor_t *)tp->tdm.monitor;

    (void)size;
    assert(NULL != tp);
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tReceived %d bytes from %d relative to taskpool %d",
                         size, src, tp->taskpool_id);
    if( src != tp->context->my_rank ) {
        tpm->stats_nb_recv_msg++;
        tpm->stats_nb_recv_bytes+=sizeof(int)+size;
    }
    switch( t ) {
    case PARSEC_TERMDET_CDA_TERMINATION_MSG_TAG:
        assert( size == sizeof(parsec_termdet_cda_empty_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tIt is a TERMINATION message");
        parsec_termdet_cda_termination_msg( src, tp, tpm );
        return PARSEC_SUCCESS;

    case PARSEC_TERMDET_CDA_CREDIT_BACK_MSG_TAG:
        assert( size == sizeof(parsec_termdet_cda_credit_carrying_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tIt is a CREDIT_BACK message with %lu credits",
                             credit_msg->value);
        parsec_termdet_cda_credit_back_msg( credit_msg, src, tp, tpm );
        return PARSEC_SUCCESS;

    case PARSEC_TERMDET_CDA_BORROW_CREDITS_TAG:
        assert( size == sizeof(parsec_termdet_cda_empty_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tIt is a BORROW_CREDIT message");
        parsec_termdet_cda_borrow_credit_msg(src, tp, tpm);
        return PARSEC_SUCCESS;

    case PARSEC_TERMDET_CDA_GIVE_CREDITS_TO_OTHER_TAG:
        assert( size == sizeof(parsec_termdet_cda_credit_carrying_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tIt is a GIVE_CREDITS message with %lu credits",
                             credit_msg->value);
        parsec_termdet_cda_give_credits_to_other_msg( credit_msg, src, tp, tpm );
        return PARSEC_SUCCESS;

    case PARSEC_TERMDET_CDA_FLUSH_TAG:
        assert( size == sizeof(parsec_termdet_cda_flush_msg_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tIs is a flush message with ID %u and %u credits",
                             flush_msg->id, flush_msg->credit);
        parsec_termdet_cda_flush_msg(flush_msg, src, tp, tpm);
        return PARSEC_SUCCESS;
    }        
    assert(0);
    return PARSEC_ERROR;
}

static void parsec_termdet_cda_monitor_taskpool(parsec_taskpool_t *tp,
                                                parsec_termdet_termination_detected_function_t cb)
{
    parsec_termdet_cda_monitor_t *tpm;
    assert(&parsec_termdet_cda_module.module == tp->tdm.module);
    
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tInstalling on taskpool %d",
                         tp->taskpool_id);

    /* We use calloc to assign 0 to all stats */
    tpm = (parsec_termdet_cda_monitor_t*)calloc(sizeof(parsec_termdet_cda_monitor_t), 1);
    
    tpm->root = 0; /* TBD: leader election?  */

    tpm->not_ready = PARSEC_TERMDET_CDA_NOT_READY;
    tpm->credit    = PARSEC_TERMDET_CDA_INITIAL_CREDIT_PER_RANK;
    tpm->committed = 0;

    parsec_atomic_rwlock_init(&tpm->lock);
    OBJ_CONSTRUCT(&tpm->delayed_messages, parsec_dequeue_t);
    tpm->flush_id = 0;
    
    tp->tdm.monitor = tpm;

    tp->tdm.counters.nb_tasks = 0;
    tp->tdm.counters.nb_pending_actions = 0;
    tp->tdm.callback = cb;

    gettimeofday(&tpm->stats_time_start, NULL);
}

static parsec_termdet_taskpool_state_t parsec_termdet_cda_taskpool_state(parsec_taskpool_t *tp)
{
    parsec_termdet_cda_monitor_t *tpm;
    parsec_termdet_taskpool_state_t ret = (parsec_termdet_taskpool_state_t)-1;
    if( tp->tdm.module == NULL )
        return PARSEC_TERM_TP_NOT_MONITORED;
    assert(tp->tdm.module == &parsec_termdet_cda_module.module);
    if( tp->tdm.monitor == PARSEC_TERMDET_CDA_TERMINATED )
        return PARSEC_TERM_TP_TERMINATED;
    tpm = tp->tdm.monitor;
    parsec_atomic_rwlock_rdlock(&tpm->lock);
    if( tpm->not_ready == PARSEC_TERMDET_CDA_NOT_READY ) {
        ret = PARSEC_TERM_TP_NOT_READY;
    } else if( tp->tdm.counters.atomic != 0 ) {
        ret = PARSEC_TERM_TP_BUSY;
    } else {
        ret = PARSEC_TERM_TP_IDLE;
    }
    parsec_atomic_rwlock_rdunlock(&tpm->lock);
    return ret;
}

static void give_credit_back_to_root(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    parsec_termdet_cda_flush_msg_t flush_msg;
    flush_msg.tag    = PARSEC_TERMDET_CDA_FLUSH_TAG;
    flush_msg.id     = tpm->flush_id;
    if( tpm->committed >= 1 ) {
        flush_msg.credit = parsec_termdet_cda_split_credit(tpm->credit, tpm->committed);
        tpm->committed--;
    } else {
        /* Handle special case of receiving credits just to purge delayed messages,
         * and they all went, so there is no real work left, so nothing is committed */
        flush_msg.credit = tpm->credit;
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tcreating flush message with id %u, committed %lu",
                         flush_msg.id, flush_msg.credit);
    parsec_comm_send_message_with_delay(tp->context->my_rank,
                                        parsec_termdet_cda_msg_tag,
                                        tp,
                                        &flush_msg, sizeof(parsec_termdet_cda_flush_msg_t),
                                        100 /* ms */);
}

static void signal_termination(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    int i;
    for(i = 0; i < parsec_termdet_cda_topology_nb_children(tp); i++) {
        parsec_termdet_cda_empty_msg_t term_msg;
        term_msg.tag = PARSEC_TERMDET_CDA_TERMINATION_MSG_TAG;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tSending TERMINATION message to rank %d",
                             parsec_termdet_cda_topology_child(tp, i));
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_cda_empty_msg_t) + sizeof(int);
        parsec_comm_send_message(parsec_termdet_cda_topology_child(tp, i),
                                     parsec_termdet_cda_msg_tag,
                                     tp,
                                     &term_msg, sizeof(parsec_termdet_cda_empty_msg_t));
    }
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tTERMINATION detected");
    parsec_termdet_cda_write_stats(tp, stdout);
    tp->tdm.monitor = PARSEC_TERMDET_CDA_TERMINATED;
    tp->tdm.callback(tp);
    free(tpm);
}

static void root_check_termination(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    if(tpm->root_missing_credit == 0 ) {
        assert(tp->tdm.counters.atomic == 0);
        assert(tpm->committed == 0);
        assert(tpm->credit == 0);
        gettimeofday(&tpm->stats_time_end, NULL);
        signal_termination(tp, tpm);
    }
}

static void check_state_workload_update(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    if( tpm->not_ready == PARSEC_TERMDET_CDA_NOT_READY )
        return;
#ifdef TERMDET_XP_IDLE_ON_NBTASKS
    if( tp->tdm.counters.nb_tasks == 0 )
#else
    if( tp->tdm.counters.atomic == 0 )
#endif
        {
            assert(tpm->credit >= 1);
            assert(tpm->committed >= 1);
            tpm->stats_nb_busy_idle++;
            gettimeofday(&tpm->stats_time_last_idle, NULL);
            if( tpm->committed == 1 )
                give_credit_back_to_root(tp, tpm);
            else {
                tpm->committed--;
            }
        }
    if( tp->context->my_rank == tpm->root ) {
        root_check_termination(tp, tpm);
    }
}

static int parsec_termdet_cda_taskpool_ready(parsec_taskpool_t *tp)
{
    parsec_termdet_cda_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );
    tpm = (parsec_termdet_cda_monitor_t*)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->lock);
    assert(tpm->not_ready == PARSEC_TERMDET_CDA_NOT_READY);
    if( tp->context->my_rank == tpm->root ) {
        tpm->root_missing_credit = tp->context->nb_nodes * PARSEC_TERMDET_CDA_INITIAL_CREDIT_PER_RANK;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tTaskpool %d becomes ready, root_missing_credit is now %lu",
                             tp->taskpool_id, tpm->root_missing_credit);

    } else {
        tpm->other_borrowing = 0;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tTaskpool %d becomes ready",
                             tp->taskpool_id);
    }
    check_state_workload_update(tp, tpm);
    if(tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED)
        parsec_atomic_rwlock_wrunlock(&tpm->lock);
    return PARSEC_SUCCESS;
}

static void commit_local_credit(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
#ifdef TERMDET_XP_IDLE_ON_NBTASKS
    if(tp->tdm.counters.nb_tasks == 0)
#else
    if(tp->tdm.counters.atomic == 0)
#endif
        {
            tpm->stats_nb_idle_busy++;
            tpm->committed++;
            tpm->flush_id++;
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tCommitting local work, committed is now %u, credit is %lu, flush id is changed to %u",
                                 tpm->committed, tpm->credit, tpm->flush_id);
            assert(tpm->credit > 0);
            assert(tpm->credit >= tpm->committed);
        }
}

static int parsec_termdet_cda_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_cda_monitor_t *tpm;
    parsec_task_counter_t ov;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );
    assert( v >= 0 );
    tpm = (parsec_termdet_cda_monitor_t *)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->lock);
    if( (int)tp->tdm.counters.nb_tasks != v) {
#ifdef TERMDET_XP_IDLE_ON_NBTASKS
        commit_local_credit(tp, tpm);
#endif
        ov = tp->tdm.counters;
        tp->tdm.counters.nb_tasks = v;
        if( ov.atomic == 0 || tp->tdm.counters.atomic == 0 )
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tCounterChange: %d/%d -> %d/%d",
                                 ov.nb_tasks, ov.nb_pending_actions,
                                 tp->tdm.counters.nb_tasks, tp->tdm.counters.nb_pending_actions);
#ifdef TERMDET_XP_IDLE_ON_NBTASKS
        check_state_workload_update(tp, tpm);
#endif
    }
    if( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->lock);
    return v;
}

static int parsec_termdet_cda_taskpool_set_nb_pa(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_cda_monitor_t *tpm;
    parsec_task_counter_t ov;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );
    assert( v >= 0 );
    tpm = (parsec_termdet_cda_monitor_t *)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->lock);
    if( (int)tp->tdm.counters.nb_pending_actions != v) {
#ifndef TERMDET_XP_IDLE_ON_NBTASKS
        commit_local_credit(tp, tpm);
#endif
        ov = tp->tdm.counters;
        tp->tdm.counters.nb_pending_actions = v;
        if( ov.atomic == 0 || tp->tdm.counters.atomic == 0 )
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tCounterChange: %d/%d -> %d/%d",
                                 ov.nb_tasks, ov.nb_pending_actions,
                                 tp->tdm.counters.nb_tasks, tp->tdm.counters.nb_pending_actions);
#ifndef TERMDET_XP_IDLE_ON_NBTASKS
        check_state_workload_update(tp, tpm);
#endif
    }
    if( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->lock);
    return v;
}

static int parsec_termdet_cda_taskpool_addto_nb_tasks_locked(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm, int v)
{
    parsec_task_counter_t ov;
    int ret;
#ifdef TERMDET_XP_IDLE_ON_NBTASKS
    commit_local_credit(tp, tpm);
#endif
    ov = tp->tdm.counters;
    tp->tdm.counters.nb_tasks += v;
    ret = tp->tdm.counters.nb_tasks;
    if( ov.atomic == 0 || tp->tdm.counters.atomic == 0 ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tCounterChange: %d/%d -> %d/%d (committed is %d)",
                             ov.nb_tasks, ov.nb_pending_actions,
                             tp->tdm.counters.nb_tasks, tp->tdm.counters.nb_pending_actions,
                             tpm->committed);
    }
    check_state_workload_update(tp, tpm);
    return ret;
}

static int parsec_termdet_cda_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_cda_monitor_t *tpm;
    int ret;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );
    if(v == 0)
        return tp->tdm.counters.nb_tasks;
    tpm = (parsec_termdet_cda_monitor_t *)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->lock);
    ret = parsec_termdet_cda_taskpool_addto_nb_tasks_locked(tp, tpm, v);
    if( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->lock);
    return ret;
}

static int parsec_termdet_cda_taskpool_addto_nb_pa_locked(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm, int v)
{
    parsec_task_counter_t ov;
    int ret;
    (void)tpm;
#ifndef TERMDET_XP_IDLE_ON_NBTASKS
    commit_local_credit(tp, tpm);
#endif
    ov = tp->tdm.counters;
    tp->tdm.counters.nb_pending_actions += v;
    ret = tp->tdm.counters.nb_pending_actions;
    if( ov.atomic == 0 || tp->tdm.counters.atomic == 0 )
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tCounterChange: %d/%d -> %d/%d",
                             ov.nb_tasks, ov.nb_pending_actions,
                             tp->tdm.counters.nb_tasks, tp->tdm.counters.nb_pending_actions);
#ifndef TERMDET_XP_IDLE_ON_NBTASKS
    check_state_workload_update(tp, tpm);
#endif
    return ret;
}

static int parsec_termdet_cda_taskpool_addto_nb_pa(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_cda_monitor_t *tpm;
    int ret;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );
    if(v == 0)
        return tp->tdm.counters.nb_pending_actions;
    tpm = (parsec_termdet_cda_monitor_t *)tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->lock);
    ret = parsec_termdet_cda_taskpool_addto_nb_pa_locked(tp, tpm, v);
    if( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->lock);
    return ret;
}

static void borrow_some_credits(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Borrowing: I am %d, root is %d\n", tp->context->my_rank, tpm->root);
    if( tpm->root == tp->context->my_rank ) {
        tpm->root_missing_credit += PARSEC_TERMDET_CDA_INITIAL_CREDIT_PER_RANK;
        tpm->credit += PARSEC_TERMDET_CDA_INITIAL_CREDIT_PER_RANK;
        tpm->flush_id++;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output,
                             "Root needs more credit, taking them, root_missing_credit is now %lu, credit is now %lu",
                             tpm->root_missing_credit, tpm->credit);
    } else {
        parsec_termdet_cda_empty_msg_t borrow_msg;
        assert(tpm->other_borrowing == 0);
        tpm->other_borrowing = 1;
        borrow_msg.tag = PARSEC_TERMDET_CDA_BORROW_CREDITS_TAG;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "Borrowing some credits from root");
        tpm->stats_nb_borrowed++;
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(int) + sizeof(borrow_msg);
        parsec_comm_send_message(tpm->root,
                                 parsec_termdet_cda_msg_tag,
                                 tp,
                                 &borrow_msg, sizeof(parsec_termdet_cda_empty_msg_t));
    }
}

static void save_message(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm, int dst_rank, parsec_remote_deps_t *remote_deps)
{
    parsec_termdet_cda_delayed_msg_t* msg = (parsec_termdet_cda_delayed_msg_t*) calloc(1, sizeof(parsec_termdet_cda_delayed_msg_t));
    (void)tp;
    OBJ_CONSTRUCT(msg, parsec_termdet_cda_delayed_msg_t);
    msg->dst_rank = dst_rank;
    msg->deps = remote_deps;
    tpm->stats_nb_delayed++;
    parsec_dequeue_push_back(&tpm->delayed_messages, (parsec_list_item_t*)msg);
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tOut of credit to send message %p to rank %d that becomes delayed",
                         remote_deps, dst_rank);
}

static void try_to_send_delayed_messages(parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    parsec_termdet_cda_delayed_msg_t* msg;
    if(tpm->credit > tpm->committed) {
        /* Root should have no delayed message */
        assert( (tp->context->my_rank != tpm->root) || parsec_dequeue_is_empty(&tpm->delayed_messages) );
        while( (tpm->committed < tpm->credit) && (NULL != (msg = (parsec_termdet_cda_delayed_msg_t*) parsec_dequeue_pop_front(&tpm->delayed_messages))) ) {
            tpm->committed++;
            tpm->flush_id++;
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tEnough credit (%lu with %d committed) to send message %p to rank %d that is removed from delayed",
                                 tpm->credit, tpm->committed, msg->deps, msg->dst_rank);
            parsec_remote_dep_send(msg->dst_rank, msg->deps);
            free(msg);
        }
        if( parsec_dequeue_is_empty(&tpm->delayed_messages) &&
            tp->tdm.counters.atomic == 0 &&
            tpm->credit > 0 ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tAll delayed messages sent and root is not active, giving back credit");
            tpm->flush_id++;
            give_credit_back_to_root(tp, tpm);
        }
    }
}

static int parsec_termdet_cda_outgoing_message_start(parsec_taskpool_t *tp,
                                                     int dst_rank,
                                                     parsec_remote_deps_t *remote_deps)
{
    int ret;
    parsec_termdet_cda_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );
    (void)dst_rank;
    (void)remote_deps;
    tpm = tp->tdm.monitor;
    parsec_atomic_rwlock_wrlock(&tpm->lock);

 try_again:
    if( tpm->committed + 1 <= tpm->credit ) {
        /* We have enough credits to commit some into this message */
        tpm->committed++;
        tpm->flush_id++;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tStart to send new message, committed becomes %d", tpm->committed);
        ret = 1;
    } else {
        /* We have to borrow more credits, and delay this message */
        assert(tpm->not_ready != PARSEC_TERMDET_CDA_NOT_READY);
        if( tp->context->my_rank == tpm->root ) {
            /* Can't read other_borrowing on root, but root is rich:
             * borrowing is immediate */
            borrow_some_credits(tp, tpm);
            goto try_again;
        } else if( tpm->other_borrowing == 0 ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tNot enough credit (%lu) to commit another emission (committed = %d), borrowing credit",
                                 tpm->credit, tpm->committed);
            borrow_some_credits(tp, tpm);
        } else {
           PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tNot enough credit (%lu) to commit another emission (committed = %d), but borrowing already in process",
                                tpm->credit, tpm->committed);
        }
        save_message(tp, tpm, dst_rank, remote_deps);
        ret = 0;
    }
    parsec_atomic_rwlock_wrunlock(&tpm->lock);
    return ret;
}

static int parsec_termdet_cda_outgoing_message_pack(parsec_taskpool_t *tp,
                                                    int dst_rank,
                                                    char *packed_buffer,
                                                    int  *position,
                                                    int buffer_size,
                                                    MPI_Comm comm)
{
    uint64_t forward;
    parsec_termdet_cda_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );

    tpm = tp->tdm.monitor;

    parsec_atomic_rwlock_wrlock(&tpm->lock);

    assert(tpm->committed > 0);
    assert(tpm->committed <= tpm->credit);
    forward = parsec_termdet_cda_split_credit(tpm->credit, tpm->committed);
    tpm->committed--;
    tpm->credit -= forward;
    tpm->flush_id++;
    
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tSending message with weight %lu, committed becomes %u, credit left is %lu",
                         forward, tpm->committed, tpm->credit);

    parsec_atomic_rwlock_wrunlock(&tpm->lock);

    MPI_Pack(&forward, 1, MPI_LONG, packed_buffer, buffer_size, position, comm);
    
    (void)dst_rank;
    (void)packed_buffer;
    (void)position;
    (void)comm;
    return PARSEC_SUCCESS;
}

static int parsec_termdet_cda_incoming_message_start(parsec_taskpool_t *tp,
                                                     int src_rank,
                                                     char *packed_buffer,
                                                     int *position,
                                                     int buffer_size,
                                                     const parsec_remote_deps_t *msg,
                                                     MPI_Comm comm)
{
    uint64_t forward;
    parsec_termdet_cda_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );

    (void)src_rank;
    (void)msg;
    
    tpm = tp->tdm.monitor;

    MPI_Unpack(packed_buffer, buffer_size, position, &forward, 1, MPI_LONG, comm);

    parsec_atomic_rwlock_wrlock(&tpm->lock);

    /* Here we take the credit. We need to ensure that this credit is not going
     * to be spent entirely before potential tasks are discovered. We cannot commit
     * onto that credit, because this fake commit might increase committed to more
     * than credit (e.g. if we received 1 only, we had 0, and this message creates a 
     * task). So, the solution is to force the process to become active by taking
     * an action that we release when the message is done. The commit inside
     * the action must succeed because there is at least 1 credit left, but
     * it will not be over-committed if we discover real tasks as the process is already
     * woken up.
     * Of course, we need to do this with the lock, hence the call to the locked version
     * of the addto_nb_pa or addto_nb_tasks */
    tpm->credit += forward;
    tpm->flush_id++;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tReceiving message with weight %lu, Increasing credit to %lu and denoting activity",
                         forward, tpm->credit);
#ifdef TERMDET_XP_IDLE_ON_NBTASKS
    parsec_termdet_cda_taskpool_addto_nb_tasks_locked(tp, tpm, 1);
#else
    parsec_termdet_cda_taskpool_addto_nb_pa_locked(tp, tpm, 1);
#endif

    /* We will check the state and deliver potential delayed messages *after*
     * all tasks are discovered, so that there is no risk of using the credit
     * that we need to compute for delayed message. */
    parsec_atomic_rwlock_wrunlock(&tpm->lock);

    return PARSEC_SUCCESS;
}

static int parsec_termdet_cda_incoming_message_end(parsec_taskpool_t *tp,
                                                   const parsec_remote_deps_t *msg)
{
    parsec_termdet_cda_monitor_t *tpm;
    (void)msg;

    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_cda_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED );
    
    tpm = tp->tdm.monitor;

    parsec_atomic_rwlock_wrlock(&tpm->lock);
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tDone receiving message");
    
#ifdef TERMDET_XP_IDLE_ON_NBTASKS
    parsec_termdet_cda_taskpool_addto_nb_tasks_locked(tp, tpm, -1);
#else
    parsec_termdet_cda_taskpool_addto_nb_pa_locked(tp, tpm, -1);
#endif
    /* We can remove the action inserted at message_start.
     * But this can detect termination, so handle the case */
    if( tp->tdm.monitor == PARSEC_TERMDET_CDA_TERMINATED ) {
        return PARSEC_SUCCESS; 
    }
    
    /* And now we can try to send the delayed messages 
     * there can be no messaged still in the delay queue if we
     * detected termination */
    try_to_send_delayed_messages(tp, tpm);
    parsec_atomic_rwlock_wrunlock(&tpm->lock);

    return PARSEC_SUCCESS;
}

static int parsec_termdet_cda_write_stats(parsec_taskpool_t *tp, FILE *fp)
{
    parsec_termdet_cda_monitor_t *tpm;
    struct timeval t1, t2;
    assert(NULL != tp->tdm.module);
    assert(&parsec_termdet_cda_module.module == tp->tdm.module);
    tpm = (parsec_termdet_cda_monitor_t *)tp->tdm.monitor;

    timersub(&tpm->stats_time_end, &tpm->stats_time_last_idle, &t1);
    timersub(&tpm->stats_time_end, &tpm->stats_time_start, &t2);
    
    fprintf(fp, "NP: %d M: CDA Rank: %d Taskpool#: %d #Transitions_Busy_to_Idle: %u #Transitions_Idle_to_Busy: %u #Times_Credit_was_Borrowed: %u #Times_Credit_was_Flushed: %u #Times_a_message_was_Delayed: %u #Times_credit_was_merged: %u #SentCtlMsg: %u #RecvCtlMsg: %u SentCtlBytes: %u RecvCtlBytes: %u WallTime: %u.%06u Idle2End: %u.%06u\n",
            tp->context->nb_nodes,
            tp->context->my_rank,
            tp->taskpool_id,
            tpm->stats_nb_busy_idle,
            tpm->stats_nb_idle_busy,
            tpm->stats_nb_borrowed,
            tpm->stats_nb_flush,
            tpm->stats_nb_delayed,
            tpm->stats_nb_credit_merge,
            tpm->stats_nb_sent_msg,
            tpm->stats_nb_recv_msg,
            tpm->stats_nb_sent_bytes,
            tpm->stats_nb_recv_bytes,
            (unsigned int)t2.tv_sec, (unsigned int)t2.tv_usec,
            (unsigned int)t1.tv_sec, (unsigned int)t1.tv_usec);

    return PARSEC_SUCCESS;
}

static void parsec_termdet_cda_termination_msg(int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    (void)src;
    
    parsec_atomic_rwlock_wrlock(&tpm->lock);
    gettimeofday(&tpm->stats_time_end, NULL);
    signal_termination(tp, tpm);
    assert(PARSEC_TERMDET_CDA_TERMINATED == tp->tdm.monitor);
}

static void parsec_termdet_cda_credit_back_msg(const parsec_termdet_cda_credit_carrying_msg_t *credit_msg, int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    (void)src;
    parsec_termdet_cda_flush_msg_t flush_msg;

    tpm = (parsec_termdet_cda_monitor_t *)tp->tdm.monitor;

    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tReceived %lu credits from %d",
                         credit_msg->value, src);

    parsec_atomic_rwlock_wrlock(&tpm->lock);

    if( tp->context->my_rank == tpm->root ) {
        tpm->root_missing_credit -= credit_msg->value;
        root_check_termination(tp, tpm);
        if( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED )
            parsec_atomic_rwlock_wrunlock(&tpm->lock);
        return;
    }

    tpm->credit += credit_msg->value;
    tpm->flush_id++;

#ifdef TERMDET_XP_IDLE_ON_NBTASKS
    if( 0 == tp->tdm.counters.nb_tasks ) {
#else
    if( 0 == tp->tdm.counters.atomic ) {
#endif
        flush_msg.tag    = PARSEC_TERMDET_CDA_FLUSH_TAG;
        flush_msg.id     = tpm->flush_id;

        if( tpm->committed >= 1 ) {
            flush_msg.credit = parsec_termdet_cda_split_credit(tpm->credit, tpm->committed);
        } else {
            /* Handle special case of receiving credits just to purge delayed messages,
             * and they all went, so there is no real work left, so nothing is committed */
            flush_msg.credit = tpm->credit;
        }
        parsec_atomic_rwlock_wrunlock(&tpm->lock);

        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tcreating flush message with id %u, committed %lu", flush_msg.id, flush_msg.credit);
        parsec_comm_send_message(tp->context->my_rank,
                                 parsec_termdet_cda_msg_tag,
                                 tp,
                                 &flush_msg, sizeof(parsec_termdet_cda_flush_msg_t));
    } else {
        assert(tpm->credit > 0);
        tpm->stats_nb_credit_merge++;
        parsec_atomic_rwlock_wrunlock(&tpm->lock);
    }
}

static void parsec_termdet_cda_borrow_credit_msg(int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    parsec_termdet_cda_credit_carrying_msg_t credit_msg;
    (void)src;
    parsec_atomic_rwlock_wrlock(&tpm->lock);
    assert( tp->context->my_rank == tpm->root );
    tpm->root_missing_credit += PARSEC_TERMDET_CDA_INITIAL_CREDIT_PER_RANK;
    credit_msg.tag = PARSEC_TERMDET_CDA_GIVE_CREDITS_TO_OTHER_TAG;
    credit_msg.value = PARSEC_TERMDET_CDA_INITIAL_CREDIT_PER_RANK;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tRoot sends %lu credits to %d, root_missing_credit is now %lu",
                         credit_msg.value, src, tpm->root_missing_credit);
    parsec_atomic_rwlock_wrunlock(&tpm->lock);

    tpm->stats_nb_sent_msg++;
    tpm->stats_nb_sent_bytes+=sizeof(int)+sizeof(parsec_termdet_cda_credit_carrying_msg_t);
    parsec_comm_send_message(src,
                             parsec_termdet_cda_msg_tag,
                             tp,
                             &credit_msg, sizeof(parsec_termdet_cda_credit_carrying_msg_t));
}

static void parsec_termdet_cda_give_credits_to_other_msg(const parsec_termdet_cda_credit_carrying_msg_t *credit_msg, int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    (void)src;
    parsec_atomic_rwlock_wrlock(&tpm->lock);
    tpm->credit += credit_msg->value;
    tpm->flush_id++;
    tpm->other_borrowing = 0;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tReceived %lu credits from root, credit is now %lu",
                         credit_msg->value, tpm->credit);
    try_to_send_delayed_messages(tp, tpm);
    if( !parsec_dequeue_is_empty(&tpm->delayed_messages) ) {
        assert(tpm->other_borrowing == 0);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tAfter sending all the message I could, there is still some messages in the queue, borrowing more credits");
        borrow_some_credits(tp, tpm);
    } else {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tDelayed messages queue is empty");
    }
    parsec_atomic_rwlock_wrunlock(&tpm->lock);
}

static void parsec_termdet_cda_flush_msg(const parsec_termdet_cda_flush_msg_t *flush_msg, int src, parsec_taskpool_t *tp, parsec_termdet_cda_monitor_t *tpm)
{
    uint64_t back;

    parsec_atomic_rwlock_wrlock(&tpm->lock);
    assert( tp->context->my_rank == src );

    if( flush_msg->id == tpm->flush_id ) {
        assert(tpm->credit >= flush_msg->credit);
        assert(tpm->credit == flush_msg->credit || tpm->committed > 1);
        back = flush_msg->credit;
    
        if( tp->context->my_rank == tpm->root ) {
            tpm->root_missing_credit -= back;
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tRoot_missing_credit is now %lu",
                                     tpm->root_missing_credit);
        } else {
            parsec_termdet_cda_credit_carrying_msg_t credit_back;
            credit_back.tag = PARSEC_TERMDET_CDA_CREDIT_BACK_MSG_TAG;
                
            credit_back.value = back;
            tpm->stats_nb_flush++;
            tpm->stats_nb_sent_msg++;
            tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_cda_credit_carrying_msg_t) + sizeof(int);
            parsec_comm_send_message(parsec_termdet_cda_topology_parent(tp),
                                     parsec_termdet_cda_msg_tag,
                                     tp,
                                     &credit_back, sizeof(parsec_termdet_cda_credit_carrying_msg_t));
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tflushing back %lu credits to root via %d, credit is now %lu, committed %d",
                                 back, parsec_termdet_cda_topology_parent(tp), tpm->credit - back, tpm->committed);
        }
        tpm->credit -= back;
    } else {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-CDA:\tflush order discarded");
    }
    if( tp->context->my_rank == tpm->root ) {
        root_check_termination(tp, tpm);
    }
    if( tp->tdm.monitor != PARSEC_TERMDET_CDA_TERMINATED )
        parsec_atomic_rwlock_wrunlock(&tpm->lock);
}
