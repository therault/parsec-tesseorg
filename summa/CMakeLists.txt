cmake_minimum_required (VERSION 2.8)


option(BUILD_SUMMA  "Build the summa library"  ON)
if(NOT BUILD_SUMMA)
  return()
endif(NOT BUILD_SUMMA)

option(SUMMA_WITH_RANDOM_TILING "testing_zsumma will generate random tile size." ON)
if(SUMMA_WITH_RANDOM_TILING)
  add_definitions(-DSUMMA_WITH_RANDOM_TILING)
endif()

LIST(APPEND CMAKE_MODULE_PATH "${DAGUE_SOURCE_DIR}/dplasma/cmake_modules/")
 find_package(COREBLAS REQUIRED)
if(NOT COREBLAS_FOUND)
  return()
endif(NOT COREBLAS_FOUND)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/testing/")

set(DPLASMA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dplasma)
set(DPLASMA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/../dplasma)

set(DAGUE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(DAGUE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)

include_directories( ${CMAKE_CURRENT_BINARY_DIR}/include )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/lib )
include_directories( ${DPLASMA_BINARY_DIR}/include )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )
include_directories( ${DAGUE_SOURCE_DIR}/data_dist/matrix )
include_directories( ${COREBLAS_INCLUDE_DIRS} )

option(SUMMA_WITH_RECURSIVE
  "Enable recursive kernels to be called when available" OFF)
if(DPLASMA_WITH_RECURSIVE)
  # TODO: This will need to be added to the futur dplasma_config.h.in
  add_definitions(-DDAGUE_HAVE_RECURSIVE)
endif()


set(EXTRA_SOURCES
  src/irregular_tiled_matrix.c
  )

include(RulesPrecisions)
include(RulesTestings)
include(RulesJDF)
include(AddDocumentedFiles)

set(generated_files "")
set(generated_jdf "")
set(generated_hdrs "")

add_documented_files(PARSEC_ALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}" ${EXTRA_SOURCES})

set(SUMMA_JDF
  lib/zsumma_TT.jdf
  lib/zsumma_TN.jdf
  lib/zsumma_NN.jdf
  lib/zsumma_NT.jdf
  )

set(SUMMA_WRAPPER_SRC
  lib/zsumma_wrapper.c
  )

precisions_rules_py(generated_jdf
  "${SUMMA_JDF}"
  PRECISIONS "${DPLASMA_PRECISIONS}")

precisions_rules_py(generated_wrappers
  "${SUMMA_WRAPPER_SRC}"
  PRECISIONS "${DPLASMA_PRECISIONS}")

jdf_rules(generated_files "${generated_jdf}")

add_documented_files( PARSEC_ALL_SRCS "${CMAKE_CURRENT_BINARY_DIR}" ${generated_wrappers} )

Add_Subdirectory(include)

add_library(summa
  ${generated_files}
  ${generated_wrappers}
  ${EXTRA_SOURCES})
add_dependencies(summa summa_includes)

target_link_libraries(summa ${DPLASMA_LIBRARIES})

if (MPI_C_FOUND)
  set_target_properties(summa PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
endif (MPI_C_FOUND)

set(SUMMA_LIBRARIES summa)

install(TARGETS summa
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES
  ${CMAKE_SOURCE_DIR}/summa/include/irregular_tiled_matrix.h
  DESTINATION include)


set(TESTING_SUMMA_SRC
  testing/testing_zsumma.c
  )


link_directories(${COREBLAS_LIBRARY_DIRS})

testings_addexec(targets "${DPLASMA_PRECISIONS}" "${TESTING_SUMMA_SRC}")
