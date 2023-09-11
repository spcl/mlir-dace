# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

add_mlir_dialect_library(
  GenericToSDFG ConvertGenericToSDFG.cpp ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SDFG/Conversion/GenericToSDFG DEPENDS
  MLIRGenericToSDFGPassIncGen)

target_link_libraries(GenericToSDFG PUBLIC MLIRIR)

target_sources(SOURCE_FILES_CPP PRIVATE ConvertGenericToSDFG.cpp)
