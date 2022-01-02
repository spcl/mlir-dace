set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
set(LIBS ${dialect_libs} ${translation_libs} MLIROptLib MLIR_SDIR)

add_llvm_executable(sdir-translate sdir-translate.cpp)
llvm_update_compile_flags(sdir-translate)

target_link_libraries(sdir-translate PRIVATE ${LIBS})
mlir_check_link_libraries(sdir-translate)
