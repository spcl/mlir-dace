# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

add_mlir_dialect_library(
  SDFGToGeneric
  ConvertSDFGToGeneric.cpp
  SymbolicParser.cpp
  OpCreators.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SDFG/Conversion/SDFGToGeneric
  DEPENDS
  MLIRSDFGToGenericPassIncGen)

target_link_libraries(SDFGToGeneric PUBLIC MLIRIR)

target_sources(SOURCE_FILES_CPP PRIVATE ConvertSDFGToGeneric.cpp
                                        SymbolicParser.cpp OpCreators.cpp)
