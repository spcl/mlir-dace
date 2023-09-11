# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

add_mlir_dialect_library(
  LinalgToSDFG ConvertLinalgToSDFG.cpp ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SDFG/Conversion/LinalgToSDFG DEPENDS
  MLIRLinalgToSDFGPassIncGen)

target_link_libraries(LinalgToSDFG PUBLIC MLIRIR)

target_sources(SOURCE_FILES_CPP PRIVATE ConvertLinalgToSDFG.cpp)
