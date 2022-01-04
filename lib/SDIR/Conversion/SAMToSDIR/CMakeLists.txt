add_mlir_dialect_library(SAMToSDIR
  ConvertSAMToSDIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SDIR/Conversion/SAMToSDIR

  DEPENDS
  MLIRSAMToSDIRPassIncGen
)

target_link_libraries(SAMToSDIR PUBLIC MLIRIR)
