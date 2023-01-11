add_mlir_translation_library(
  MLIRTargetSDFG
  registration.cpp
  translateToSDFG.cpp
  liftToPython.cpp
  JsonEmitter.cpp
  Node.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/SDFG
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRTranslateLib
  SDFG_UTILS)

target_sources(
  SOURCE_FILES_CPP PRIVATE registration.cpp translateToSDFG.cpp
                           liftToPython.cpp JsonEmitter.cpp Node.cpp)
