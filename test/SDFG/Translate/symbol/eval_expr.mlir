// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py
module  {
  sdir.sdfg {entry = @state_0} @kernel_2mm(%arg1: !sdir.memlet<index>) {
    sdir.alloc_symbol("idx")

    sdir.state @state_0 {
      %2 = sdir.sym("idx") : index
      sdir.store %2, %arg1[] : index -> !sdir.memlet<index>

    }
  }
}
