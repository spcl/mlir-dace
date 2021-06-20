add_mlir_dialect_library(
        MLIRSDIR
        SDIRDialect.cpp
        SDIROps.cpp

        ADDITIONAL_HEADER_DIRS {PROJECT_SOURCE_DIR}/include/SDIR
        DEPENDS MLIRSDIROpsIncGen
	LINK_LIBS PUBLIC MLIRIR
)
