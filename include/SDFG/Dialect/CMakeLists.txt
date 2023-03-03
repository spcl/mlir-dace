add_mlir_dialect(Ops sdfg)
add_mlir_doc(Dialect Dialect SDFG/ -gen-dialect-doc)
add_mlir_doc(Ops Ops SDFG/ -gen-op-doc)
target_sources(SOURCE_FILES_H PRIVATE Dialect.h)
