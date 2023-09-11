# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS ${dialect_libs} ${translation_libs} MLIROptLib MLIR_SDFG)

add_llvm_executable(sdfg-translate sdfg-translate.cpp)
llvm_update_compile_flags(sdfg-translate)

target_link_libraries(sdfg-translate PRIVATE ${LIBS})
mlir_check_link_libraries(sdfg-translate)

target_sources(SOURCE_FILES_CPP PRIVATE sdfg-translate.cpp)
