// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

module  {
  sdfg.sdfg {entry = @state_0} @kernel_2mm (%arg9: !sdfg.array<sym("s_2")x1200xf64>) {
    sdfg.state @state_0 {

      sdfg.sdfg {entry = @init_4} @sdfg_3(%arg0: !sdfg.array<sym("s_2")x1200xf64>) -> (%arg9: !sdfg.array<sym("s_2")x1200xf64>) {
        sdfg.state @init_4 {}
      }
    }
  }
}
