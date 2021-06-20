get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIROptLib MLIR_SDIR)

add_llvm_executable(sdir-opt sdir-opt.cpp)
llvm_update_compile_flags(sdir-opt)

target_link_libraries(sdir-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(sdir-opt)
