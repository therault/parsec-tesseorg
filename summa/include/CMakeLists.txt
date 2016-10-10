include(RulesPrecisions)

# reset variables
set(generated_headers "")

### generate the dplasma headers for all possible precisions
precisions_rules_py(generated_headers "summa_z.h" PRECISIONS "z;c;d;s")

add_custom_target(summa_includes ALL SOURCES ${generated_headers} )

foreach(generated_header ${generated_headers})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${generated_header} DESTINATION include)
endforeach()

