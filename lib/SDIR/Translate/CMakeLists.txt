add_mlir_translation_library(
  MLIRTargetSDFG
  registration.cpp
  translateToSDFG.cpp
  JsonEmitter.cpp

  ADDITIONAL_HEADER_DIRS 
  ${PROJECT_SOURCE_DIR}/include/SDIR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTranslation
)
