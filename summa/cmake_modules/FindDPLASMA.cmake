
# - Find PARSEC library
# This module finds an installed  library that implements PaRSEC
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module is controled by the following variables:
#  DPLASMA_DIR - path to look for PaRSEC
#  DPLASMA_PKG_DIR - path to look for the parsec.pc pkgconfig file
#  DPLASMA_FOUND - set to true if a library implementing the PLASMA interface
#    is found
#  DPLASMA_INCLUDE_DIRS - include directories
#  DPLASMA_LIBRARIES - uncached list of libraries (using full path name) to
#    link against to use PaRSEC
#  DPLASMA_STATIC  if set on this determines what kind of linkage we do (static)
#  DPLASMA_VENDOR  if set checks only the specified vendor, if not set checks
#     all the possibilities
##########

# First we try to use pkg-config to find what we're looking for
# in the directory specified by the DPLASMA_DIR or DPLASMA_PKG_DIR
if( DPLASMA_DIR )
  if( DPLASMA_PKG_DIR )
    message(STATUS "DPLASMA_DIR and DPLASMA_PKG_DIR are set at the same time; ${DPLASMA_DIR} overrides ${DPLASMA_PKG_DIR}.")
  endif()
endif( DPLASMA_DIR )
include(FindPkgConfig)
set(ENV{PKG_CONFIG_PATH} "${DPLASMA_PKG_DIR}:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(DPLASMA dplasma)

if( NOT DPLASMA_FOUND )
  if( DPLASMA_DIR )
    if( NOT DPLASMA_INCLUDE_DIRS )
      set(DPLASMA_INCLUDE_DIRS "${DPLASMA_DIR}/include")
    endif()
  else( DPLASMA_DIR )
    if( DPLASMA_FIND_REQUIRED )
      message(FATAL_ERROR "DPLASMA: NOT FOUND. pkg-config not available. You need to provide DPLASMA_DIR.")
    endif()
  endif( DPLASMA_DIR )

  include(CheckIncludeFiles)
  check_include_files(dplasma.h FOUND_DPLASMA_INCLUDE)
  include_directories( ${DPLASMA_INCLUDE_DIRS} )
  if( NOT FOUND_DPLASMA_INCLUDE )
    if( DPLASMA_FIND_REQUIRED )
      message(FATAL_ERROR "dplasma.h: NOT FOUND in ${DPLASMA_INCLUDE_DIRS}.")
    endif()
  endif( NOT FOUND_DPLASMA_INCLUDE )

endif( NOT DPLASMA_FOUND )

mark_as_advanced(DPLASMA_DIR DPLASMA_PKG_DIR DPLASMA_LIBRARY DPLASMA_LIBRARIES DPLASMA_INCLUDE_DIRS)
set(DPLASMA_DIR "${DPLASMA_DIR}" CACHE PATH "Location of the DPLASMA library" FORCE)
set(DPLASMA_PKG_DIR "${DPLASMA_PKG_DIR}" CACHE PATH "Location of the DPLASMA pkg-config description file" FORCE)
set(DPLASMA_INCLUDE_DIRS "${DPLASMA_INCLUDE_DIRS}" CACHE PATH "DPLASMA include directories" FORCE)
set(DPLASMA_LIBRARIES "${DPLASMA_LIBRARIES}" CACHE STRING "libraries to link with DPLASMA" FORCE)

#find_package_message(DPLASMA
#    "Found DPLASMA: ${DPLASMA_LIB}
#        DPLASMA_INCLUDE_DIRS      = [${DPLASMA_INCLUDE_DIRS}]
#        DPLASMA_LIBRARIES         = [${DPLASMA_LIBRARIES}]
#      "[${DPLASMA_INCLUDE_DIRS}][${DPLASMA_LIBRARIES}]")
