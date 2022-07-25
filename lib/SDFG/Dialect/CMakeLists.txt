add_mlir_dialect_library(
        MLIR_SDFG
        Dialect.cpp
        Ops.cpp

        ADDITIONAL_HEADER_DIRS 
        ${PROJECT_SOURCE_DIR}/include/SDFG/Dialect

        DEPENDS MLIROpsIncGen

	LINK_LIBS PUBLIC 
        MLIRIR
        SDFG_UTILS
)
