/*
 * Copyright (c) 2018      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include "parsec/parsec_config.h"
#include "parsec.h"
#include "parsec/parsec_internal.h"
#include "parsec/remote_dep.h"

#include "parsec/mca/termdet/termdet.h"
#include "parsec/mca/termdet/cda/termdet_cda.h"

/*
 * Local function
 */
static int termdet_cda_component_query(mca_base_module_t **module, int *priority);
static int termdet_cda_component_close(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */
const parsec_termdet_base_component_t parsec_termdet_cda_component = {

    /* First, the mca_component_t struct containing meta information
       about the component itself */

    {
        PARSEC_TERMDET_BASE_VERSION_2_0_0,

        /* Component name and version */
        "cda",
        PARSEC_VERSION_MAJOR,
        PARSEC_VERSION_MINOR,

        /* Component open and close functions */
        NULL, /*< No open: termdet_cda is always available, no need to check at runtime */
        termdet_cda_component_close,
        termdet_cda_component_query, 
        /*< specific query to return the module and add it to the list of available modules */
        NULL, /*< No register: no parameters to the absolute priority component */
        "", /*< no reserve */
    },
    {
        /* The component has no metada */
        MCA_BASE_METADATA_PARAM_NONE,
        "", /*< no reserve */
    }
};

mca_base_component_t *termdet_cda_static_component(void)
{
    return (mca_base_component_t *)&parsec_termdet_cda_component;
}

/* set to 1 when the callback is registered -- workaround current MCA interface limitation */
static int parsec_termdet_cda_msg_cb_registered = 0;
int parsec_termdet_cda_msg_tag = -127; 

static int termdet_cda_component_query(mca_base_module_t **module, int *priority)
{
    /* module type should be: const mca_base_module_t ** */
    void *ptr = (void*)&parsec_termdet_cda_module;
    *priority = 10;
    *module = (mca_base_module_t *)ptr;

    if( 0 == parsec_termdet_cda_msg_cb_registered ) {
        parsec_comm_register_callback(&parsec_termdet_cda_msg_tag,
                                      PARSEC_TERMDET_CDA_MAX_MSG_SIZE,
                                      1,
                                      parsec_termdet_cda_msg_dispatch);
        parsec_termdet_cda_msg_cb_registered = 1;
    }
    
    return MCA_SUCCESS;
}

static int termdet_cda_component_close()
{
    if( 1 == parsec_termdet_cda_msg_cb_registered ) {
        parsec_comm_unregister_callback(parsec_termdet_cda_msg_tag);
        parsec_termdet_cda_msg_tag = -127;
        parsec_termdet_cda_msg_cb_registered = 0;
    }
    return MCA_SUCCESS;
}

