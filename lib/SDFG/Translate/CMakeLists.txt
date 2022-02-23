add_mlir_translation_library(
  MLIRTargetSDFG
  registration.cpp
  translateToSDFG.cpp
  JsonEmitter.cpp
  Node.cpp

  ADDITIONAL_HEADER_DIRS 
  ${PROJECT_SOURCE_DIR}/include/SDFG

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTranslation
  SDFG_UTILS
)
