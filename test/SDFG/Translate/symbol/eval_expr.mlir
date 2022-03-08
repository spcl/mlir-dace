// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

module  {
  sdfg.sdfg {entry = @state_0} @kernel_2mm(%arg1: !sdfg.array<index>) {
    sdfg.alloc_symbol("idx")

    sdfg.state @state_0 {
      %2 = sdfg.sym("idx") : index
      sdfg.store %2, %arg1[] : index -> !sdfg.array<index>
    }
  }
}
