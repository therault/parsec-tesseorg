/*
 * Copyright (c) 2018      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 */


/**
 * @file
 *
 * Efficient Delay-Optimal Distributed Termination Detection Algorithm (EDOD)
 *   (see N. R. Mahapatra and S. Dutt. An efficient delay-optimal distributed termination detection algorithm. J. Parallel Distributed Computing, 67(10):1047 – 1066, 2007.)
 *
 */

#ifndef MCA_TERMDET_EDOD_H
#define MCA_TERMDET_EDOD_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/termdet/termdet.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_termdet_base_component_t parsec_termdet_edod_component;
PARSEC_DECLSPEC extern const parsec_termdet_module_t parsec_termdet_edod_module;

extern int parsec_termdet_edod_msg_tag; 
extern int parsec_termdet_edod_msg_dispatch(int src, parsec_taskpool_t *tp, const void *msg, size_t size);

typedef enum {
    PARSEC_TERMDET_EDOD_TERMINATION_MSG,
    PARSEC_TERMDET_EDOD_STOP_MSG,
    PARSEC_TERMDET_EDOD_ACKNOWLEDGE_MSG,
    PARSEC_TERMDET_EDOD_RESUME_MSG
} parsec_termdet_edod_msg_type_t;

typedef struct {
    parsec_termdet_edod_msg_type_t msg_type;
    int first;
    int second;
} parsec_termdet_edod_tworanks_msg_t;

typedef struct {
    parsec_termdet_edod_msg_type_t msg_type;
} parsec_termdet_edod_empty_msg_t;

#define PARSEC_TERMDET_EDOD_MAX_MSG_SIZE sizeof(parsec_termdet_edod_tworanks_msg_t)

/* static accessor */
mca_base_component_t *termdet_edod_static_component(void);

END_C_DECLS
#endif /* MCA_TERMDET_EDOD_H */

