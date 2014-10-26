/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DATA_H_HAS_BEEN_INCLUDED
#define DATA_H_HAS_BEEN_INCLUDED

#include "dague.h"
#include "dague/class/dague_object.h"
#include "dague/class/list_item.h"
#include "dague/types.h"

struct dague_context_s;

typedef uint8_t dague_data_coherency_t;
#define    DATA_COHERENCY_INVALID   ((dague_data_coherency_t)0x0)
#define    DATA_COHERENCY_OWNED     ((dague_data_coherency_t)0x1)
#define    DATA_COHERENCY_EXCLUSIVE ((dague_data_coherency_t)0x2)
#define    DATA_COHERENCY_SHARED    ((dague_data_coherency_t)0x4)

typedef uint8_t dague_data_status_t;
#define    DATA_STATUS_NOT_TRANSFER        ((dague_data_coherency_t)0x0)
#define    DATA_STATUS_UNDER_TRANSFER        ((dague_data_coherency_t)0x1)
#define    DATA_STATUS_COMPLETE_TRANSFER     ((dague_data_coherency_t)0x2)

typedef uint8_t dague_data_flag_t;
#define DAGUE_DATA_FLAG_ARENA     ((dague_data_flag_t)0x01)
#define DAGUE_DATA_FLAG_TRANSIT   ((dague_data_flag_t)0x02)

/**
 * This structure represent a device copy of a dague_data_t.
 */
struct dague_data_copy_s {
    dague_list_item_t         super;

    int8_t                    device_index;         /**< Index in the original->device_copies array */
    dague_data_flag_t         flags;
    dague_data_coherency_t    coherency_state;
    /* int8_t */

    int32_t                   readers;

    uint32_t                  version;

    struct dague_data_copy_s *older;                 /**< unused yet */
    dague_data_t             *original;
    struct dague_arena_chunk_s      *arena_chunk;           /**< If this is an arena-based data, keep
                                                      *   the chunk pointer here, to avoid
                                                      *   risky pointers arithmetic (pointers mis-alignment
                                                      *   depending on many parameters) */
    void                     *device_private;        /**< The pointer to the device-specific data.
                                                      *   Overlay data distributions assume that arithmetic
                                                      *   can be done on these pointers. */
    dague_data_status_t      data_transfer_status;   /** three status */
    struct dague_execution_context_t *push_task;            /** the task who actually do the PUSH */
};
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_data_copy_t);

/**
 * Initialize the DAGuE data infrastructure
 */
DAGUE_DECLSPEC int dague_data_init(struct dague_context_s* context);
DAGUE_DECLSPEC int dague_data_fini(struct dague_context_s* context);

DAGUE_DECLSPEC dague_data_copy_t*
dague_data_get_copy(dague_data_t* data, uint32_t device);

DAGUE_DECLSPEC dague_data_copy_t*
dague_data_copy_new(dague_data_t* data, uint8_t device);
/**
 * Decrease the refcount of this copy of the data. If the refcount reach
 * 0 the upper level is in charge of cleaning up and releasing all content
 * of the copy.
 */
DAGUE_DECLSPEC void dague_data_copy_release(dague_data_copy_t* copy);

/**
 * Return the device private pointer for a datacopy.
 */
DAGUE_DECLSPEC void* dague_data_copy_get_ptr(dague_data_copy_t* data);

/**
 * Allocate a new data structure set to INVALID and no attached copies.
 */
DAGUE_DECLSPEC dague_data_t* dague_data_new(void);

/**
 * Force the release of a data (which become unavailable for further uses).
 */
DAGUE_DECLSPEC void dague_data_delete(dague_data_t* data);

/**
 * Attach a new copy corresponding to the specified device to a data. If a copy
 * for the device is already attached, nothing will be done and an error code
 * will be returned.
 */
DAGUE_DECLSPEC int
dague_data_copy_attach(dague_data_t* data,
                       dague_data_copy_t* copy,
                       uint8_t device);
DAGUE_DECLSPEC int
dague_data_copy_detach(dague_data_t* data,
                       dague_data_copy_t* copy,
                       uint8_t device);

DAGUE_DECLSPEC int
dague_data_transfer_ownership_to_copy(dague_data_t* map,
                                      uint8_t device,
                                      uint8_t access_mode);
DAGUE_DECLSPEC void dague_dump_data_copy(dague_data_copy_t* copy);
DAGUE_DECLSPEC void dague_dump_data(dague_data_t* copy);

#endif  /* DATA_H_HAS_BEEN_INCLUDED */
