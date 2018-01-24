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
 * Credit Distribution Algorithm Algorithm, SC'18 paper variant
 *   (see TBD)
 *
 */

#ifndef MCA_TERMDET_CDA_H
#define MCA_TERMDET_CDA_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/termdet/termdet.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_termdet_base_component_t parsec_termdet_cda_component;
PARSEC_DECLSPEC extern const parsec_termdet_module_t parsec_termdet_cda_module;

extern int parsec_termdet_cda_msg_tag; 
extern int parsec_termdet_cda_msg_dispatch(int src, parsec_taskpool_t *tp, const void *msg, size_t size);

typedef enum {
    PARSEC_TERMDET_CDA_CREDIT_BACK_MSG_TAG,
    PARSEC_TERMDET_CDA_TERMINATION_MSG_TAG,
    PARSEC_TERMDET_CDA_BORROW_CREDITS_TAG,
    PARSEC_TERMDET_CDA_GIVE_CREDITS_TO_OTHER_TAG,
    PARSEC_TERMDET_CDA_FLUSH_TAG
} parsec_termdet_cda_msg_type_t;

typedef struct {
    parsec_termdet_cda_msg_type_t tag;
} parsec_termdet_cda_empty_msg_t;

typedef struct {
    parsec_termdet_cda_msg_type_t tag;
    uint64_t value;
} parsec_termdet_cda_credit_carrying_msg_t;

typedef struct {
    parsec_termdet_cda_msg_type_t tag;
    uint32_t id;
    uint64_t credit;
} parsec_termdet_cda_flush_msg_t;

#define PARSEC_TERMDET_CDA_MAX_MSG_SIZE sizeof(parsec_termdet_cda_flush_msg_t)

/* static accessor */
mca_base_component_t *termdet_cda_static_component(void);

END_C_DECLS
#endif /* MCA_TERMDET_CDA_H */

