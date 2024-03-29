# Copyright (c) 2021-2023, Scalable Parallel Computing Lab, ETH Zurich

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIR_SDFG
    GenericToSDFG
    LinalgToSDFG
    SDFGToGeneric)

add_llvm_executable(sdfg-opt sdfg-opt.cpp)
llvm_update_compile_flags(sdfg-opt)

target_link_libraries(sdfg-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(sdfg-opt)

target_sources(SOURCE_FILES_CPP PRIVATE sdfg-opt.cpp)
