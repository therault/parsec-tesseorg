include(ParsecCompilePTG)

parsec_addtest(C project "main.c;tree_dist.c")
target_ptg_sources(project PRIVATE "project.jdf;walk.jdf")
target_include_directories(project PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(project PRIVATE Threads::Threads m)

parsec_addtest(C rwalk "rand_walk.c")
target_ptg_sources(rwalk PRIVATE "random_walk.jdf")
target_include_directories(rwalk PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)

parsec_addtest(C irtest "ir_test.c;osu_allreduce.c;osu_coll.c;tree_dist.c")
target_ptg_sources(irtest PRIVATE "random_walk.jdf;project.jdf")
target_compile_definitions(irtest PUBLIC parsec_project_new=parsec_project_dyn_new)
target_include_directories(irtest PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)

parsec_addtest(C project_dyn "main.c;tree_dist.c")
target_ptg_sources(project_dyn PRIVATE "project.jdf")
target_compile_definitions(project_dyn PUBLIC parsec_project_new=parsec_project_dyn_new)
target_include_directories(project_dyn PRIVATE $<$<NOT:${PARSEC_BUILD_INPLACE}>:${CMAKE_CURRENT_SOURCE_DIR}>)

add_test(unit_haar_tree_shm ${SHM_TEST_CMD_LIST} ./project -x)
if( MPI_C_FOUND )
  add_test(unit_haar_tree_mpi ${MPI_TEST_CMD_LIST} 4 ./project -x)
  set_tests_properties(unit_haar_tree_mpi PROPERTIES DEPENDS launch_mpi)
endif( MPI_C_FOUND )

