add_mlir_dialect_library(
        MLIR_SDIR
        Dialect.cpp
        Ops.cpp

        ADDITIONAL_HEADER_DIRS {PROJECT_SOURCE_DIR}/include/SDIR/Dialect
        DEPENDS MLIROpsIncGen
	LINK_LIBS PUBLIC MLIRIR
)
