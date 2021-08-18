add_mlir_dialect_library(
        MLIR_SDIR
        SDIR_Dialect.cpp
        SDIR_Ops.cpp

        ADDITIONAL_HEADER_DIRS {PROJECT_SOURCE_DIR}/include/SDIR
        DEPENDS MLIRSDIR_OpsIncGen
	LINK_LIBS PUBLIC MLIRIR
)
