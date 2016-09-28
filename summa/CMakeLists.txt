cmake_minimum_required (VERSION 2.8)


option(BUILD_SUMMA  "Build the summa library"  ON)
if(NOT BUILD_SUMMA)
  return()
endif(NOT BUILD_SUMMA)


LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")


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
include(RulesJDF)
include(AddDocumentedFiles)

set(generated_files "")
set(generated_jdf "")

add_documented_files(PARSEC_ALL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}" ${EXTRA_SOURCES})

set(SUMMA_JDF
  lib/zgemm_TT.jdf
  lib/zgemm_TN.jdf
  lib/zgemm_NN.jdf
  lib/zgemm_NT.jdf
  )


precisions_rules_py(generated_jdf
                    "${SUMMA_JDF}"
                    PRECISIONS "${DPLASMA_PRECISIONS}")

jdf_rules(generated_files "${generated_jdf}")

message( "--- ${SUMMA_JDF} --- ${DPLASMA_PRECISIONS} --- ${generated_jdf} --- ${generated_files} ---")

set(SUMMA_WRAPPER_SRC
  zgemm_wrapper.c
  )

precisions_rules_py(generated_wrappers
  "${SUMMA_WRAPPER_SRC}"
  PRECISIONS "${DPLASMA_PRECISIONS}")

add_documented_files( PARSEC_ALL_SRCS "${CMAKE_CURRENT_BINARY_DIR}/" ${generated_wrappers} )

add_library(summa
  ${generated_files}
  ${generated_wrappers}
  ${EXTRA_SOURCES})

target_link_libraries(summa
  ${DPLASMA_LIBRARIES}
)

if (MPI_C_FOUND)
  set_target_properties(summa PROPERTIES COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
endif (MPI_C_FOUND)

install(TARGETS summa
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES
  ${CMAKE_SOURCE_DIR}/include/irregular_tiled_matrix.h
  DESTINATION include)





set(TESTING_SUMMA_SRC
  testing/testing_summa.c
  )

include(RulesTestings.cmake)

testings_addexec(targets "${DPLASMA_PRECISIONS}" "${TESTING_SUMMA_SRC}")
