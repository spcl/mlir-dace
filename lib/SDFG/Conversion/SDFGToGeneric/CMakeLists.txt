add_mlir_dialect_library(
  SDFGToGeneric ConvertSDFGToGeneric.cpp ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SDFG/Conversion/SDFGToGeneric DEPENDS
  MLIRSDFGToGenericPassIncGen)

target_link_libraries(SDFGToGeneric PUBLIC MLIRIR)

target_sources(SOURCE_FILES_CPP PRIVATE ConvertSDFGToGeneric.cpp)
