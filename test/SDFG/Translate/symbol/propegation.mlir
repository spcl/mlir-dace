// RUN: sdfg-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py
module  {
  sdfg.sdfg {entry = @state_0} @kernel_2mm (%arg9: !sdfg.memlet<sym("s_2")x1200xf64>) {
    sdfg.state @state_0 {

      sdfg.sdfg {entry = @init_4} @sdfg_3(%arg0: !sdfg.memlet<sym("s_2")x1200xf64>) {
        sdfg.state @init_4 {}
      }

      sdfg.call @sdfg_3(%arg9) : (!sdfg.memlet<sym("s_2")x1200xf64>) -> ()
    }
  }
}
