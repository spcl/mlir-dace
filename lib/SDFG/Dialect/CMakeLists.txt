# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

add_mlir_dialect_library(
  MLIR_SDFG
  Dialect.cpp
  Ops.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SDFG/Dialect
  DEPENDS
  MLIROpsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  SDFG_UTILS)

target_sources(SOURCE_FILES_CPP PRIVATE Dialect.cpp Ops.cpp)
