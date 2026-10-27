target_sources(strumpack
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/HODLROptions.hpp
  ${CMAKE_CURRENT_LIST_DIR}/HODLROptions.cpp)

install(FILES
  HODLROptions.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HODLR)


if(STRUMPACK_USE_BPACK)
  target_sources(strumpack
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/HODLRMatrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRMatrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ButterflyMatrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/ButterflyMatrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRWrapper.hpp
    ${CMAKE_CURRENT_LIST_DIR}/HODLRWrapper.cpp)

  install(FILES
    HODLRMatrix.hpp
    ButterflyMatrix.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HODLR)
endif()

if(STRUMPACK_USE_HODLR_GPU_EVALUATOR)
  # Isolate CUDA compilation from the ordinary host compiler's flags.
  add_library(strumpack_hodlr_gpu_evaluator OBJECT GPUFrontEvaluator.cu)
  set_target_properties(strumpack_hodlr_gpu_evaluator PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_link_libraries(strumpack_hodlr_gpu_evaluator PRIVATE CUDA::cudart)
  target_compile_definitions(strumpack_hodlr_gpu_evaluator PRIVATE
    $<$<BOOL:${STRUMPACK_USE_HODLR_GPU_RECTANGULAR}>:STRUMPACK_USE_HODLR_GPU_RECTANGULAR>)
  if(TARGET ButterflyPACK::butterflypack)
    target_link_libraries(strumpack_hodlr_gpu_evaluator PRIVATE ButterflyPACK::butterflypack)
  else()
    target_link_libraries(strumpack_hodlr_gpu_evaluator PRIVATE ButterflyPACK::dbutterflypack)
  endif()
  target_sources(strumpack PRIVATE $<TARGET_OBJECTS:strumpack_hodlr_gpu_evaluator>)
  install(FILES GPUFrontEvaluator.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/HODLR)
endif()
