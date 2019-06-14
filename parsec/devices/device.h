/*
 * Copyright (c) 2013-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_DEVICE_H_HAS_BEEN_INCLUDED
#define PARSEC_DEVICE_H_HAS_BEEN_INCLUDED

#include "parsec/class/list_item.h"
#if defined(PARSEC_PROF_TRACE)
#include "parsec/profiling.h"
#endif  /* defined(PARSEC_PROF_TRACE) */
#include "parsec/runtime.h"
#include "parsec/data_distribution.h"

#define PARSEC_DEV_NONE       ((uint8_t)    0x00)
#define PARSEC_DEV_CPU        ((uint8_t)(1 << 0))
#define PARSEC_DEV_RECURSIVE  ((uint8_t)(1 << 1))
#define PARSEC_DEV_CUDA       ((uint8_t)(1 << 2))
#define PARSEC_DEV_INTEL_PHI  ((uint8_t)(1 << 3))
#define PARSEC_DEV_OPENCL     ((uint8_t)(1 << 4))
#define PARSEC_DEV_ALL        ((uint8_t)    0x1f)

#define PARSEC_DEV_DATA_ADVICE_PREFETCH              ((int) 0x01)
#define PARSEC_DEV_DATA_ADVICE_PREFERRED_DEVICE      ((int) 0x02)
#define PARSEC_DEV_DATA_ADVICE_WARMUP                ((int) 0x03)

typedef struct parsec_device_s parsec_device_t;

typedef int   (*parsec_device_init_f)(parsec_device_t*);
typedef int   (*parsec_device_fini_f)(parsec_device_t*);
typedef int   (*parsec_device_taskpool_register_f)(parsec_device_t*, parsec_taskpool_t*);
typedef int   (*parsec_device_taskpool_unregister_f)(parsec_device_t*, parsec_taskpool_t*);
typedef int   (*parsec_device_memory_register_f)(parsec_device_t*, parsec_data_collection_t*, void*, size_t);
typedef int   (*parsec_device_memory_unregister_f)(parsec_device_t*, parsec_data_collection_t*, void*);
typedef int   (*parsec_device_memory_release_f)(parsec_device_t*);
typedef int   (*parsec_device_data_advise_f)(parsec_device_t*, parsec_data_t *, int);
typedef void* (*parsec_device_find_function_f)(parsec_device_t*, char*);

struct parsec_device_s {
    parsec_list_item_t item;

    /* Device Management Functions */
    parsec_device_fini_f                device_fini;
    parsec_device_taskpool_register_f   device_taskpool_register;
    parsec_device_taskpool_unregister_f device_taskpool_unregister;
    parsec_device_memory_register_f     device_memory_register;
    parsec_device_memory_unregister_f   device_memory_unregister;
    parsec_device_memory_release_f      device_memory_release;
    parsec_device_data_advise_f         device_data_advise;
    parsec_device_find_function_f       device_find_function;

    struct parsec_context_s* context;  /**< The PaRSEC context this device belongs too */
    char* name;  /**< Simple identified for the device */
    uint64_t transferred_data_in;
    uint64_t transferred_data_out;
    uint64_t d2d_transfer;
    uint64_t required_data_in;
    uint64_t required_data_out;
    uint64_t executed_tasks;
    uint64_t nb_data_faults;
    float device_hweight;  /**< Number of half precision operations per second */
    float device_sweight;  /**< Number of single precision operations per second */
    float device_dweight;  /**< Number of double precision operations per second */
    float device_tweight;  /**< Number of tensor operations per second */
#if defined(PARSEC_PROF_TRACE)
    parsec_thread_profiling_t *profiling;
#endif  /* defined(PROFILING) */
    uint8_t device_index;
    uint8_t type;
};

BEGIN_C_DECLS

extern uint32_t parsec_nb_devices;
extern int parsec_device_output;
extern parsec_atomic_lock_t parsec_devices_mutex;
/**
 * Temporary variables used for load-balancing purposes.
 */
extern float *parsec_device_load;
extern float *parsec_device_sweight;
extern float *parsec_device_dweight;

/**
 * Initialize the internal structures for managing external devices such as
 * accelerators and GPU. Memory nodes can as well be managed using the same
 * mechnism.
 */
extern int parsec_devices_init(parsec_context_t*);

/**
 * The runtime will shutdown, all internal structures have to be destroyed.
 */
extern int parsec_devices_fini(parsec_context_t*);

/**
 * Parse the list of potential devices and see which one would succesfully load
 * and initialize in the current environment.
 */
extern int parsec_devices_select(parsec_context_t*);

/**
 * This call mark the end of the configuration step, no devices can be registered
 * after this point. This gives a reference point regarding the number of available
 * (supported) devices.
 */
extern int parsec_devices_freeze(parsec_context_t*);
/**
 * Return a positive value if the devices configurations has been freezed by a call
 * to parsec_devices_freeze().
 */
extern int parsec_devices_freezed(parsec_context_t*);

/**
 * Reset the load of all the devices to force a reconsideration of the load balance
 */
PARSEC_DECLSPEC void parsec_devices_reset_load(parsec_context_t *context);

/**
 * Declare a new device with the runtime. The device will later provide a list
 * of supported operations.
 */
PARSEC_DECLSPEC int parsec_devices_add(parsec_context_t*, parsec_device_t*);

/**
 * Retrieve a pointer to the registered device using the provided index.
 */
PARSEC_DECLSPEC parsec_device_t* parsec_devices_get(uint32_t);

/**
 * Remove the device from the list of enabled devices. All data residing on the
 * device will be copied back on the main memory (or the memory declared as the
 * originator of the data), and all tasks owned by the device will be discarded
 * and moved back into the main scheduling mechanism.
 */
PARSEC_DECLSPEC int parsec_devices_remove(parsec_device_t* device);

/**
 * Dump and reset the current device statistics.
 */
PARSEC_DECLSPEC void parsec_devices_dump_and_reset_statistics(parsec_context_t* parsec_context);

/**
 * Returns the number of devices currently registered with the runtime. This
 * number can change until parsec_devices_freeze() is called, fact that mark the
 * end of the configuration stage.
 */
static inline int parsec_devices_enabled(void)
{
    return parsec_nb_devices;
}

/**
 * Restrict the device type that can be used to execute the taskpool
 */
PARSEC_DECLSPEC void parsec_devices_taskpool_restrict( parsec_taskpool_t *tp,
                                                       uint8_t            devices_type );

/**
 * Release all additional memory allocated on device.
 *
 * Device 0 (CPU) does not release the memory allocated on it,
 * only devices with local memory (e.g. GPUs) release temporary
 * buffers. This is used to start with an empty cache.
 */
PARSEC_DECLSPEC int parsec_devices_release_memory(void);

/**
 * Provides hints to a device about data
 *
 * Possible advices are PARSEC_DEV_DATA_ADVICE_*
 *   PREFETCH: a copy corresponding to the data should be prefetch
 *             on the device
 *   PREFERRED_DEVICE: this device is the preferred device to own
 *                     the data (may be used when selecting given
 *                     devices)
 *   WARMUP: if the device uses a cache policy, this tells that
 *           the data should be considered as recently used by
 *           the call.
 *
 * The advice may be ignored by the device. Each device sets their
 * policies wrt prefetching and caching through MCA parameters.
 */
PARSEC_DECLSPEC int parsec_advise_data_on_device(parsec_data_t *data, int device, int advice);

/**
 * Find a function is a set of shared libraries specified in paths. If a path
 * points to a directory, the libname is added to pinpoint to the expected shared
 * library. If no functions has been found on the paths the scope of the current
 * application is searched for the function_name. Upon success the pointer to the
 * function is returned, otherwise NULL.
 */
PARSEC_DECLSPEC void*
parsec_device_find_function(const char* function_name,
                            const char* libname,
                            const char* paths[]);

END_C_DECLS

#endif  /* PARSEC_DEVICE_H_HAS_BEEN_INCLUDED */
