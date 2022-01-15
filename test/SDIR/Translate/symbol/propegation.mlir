// RUN: sdir-translate --mlir-to-sdfg %s | python %S/../import_translation_test.py
module  {
  sdir.sdfg {entry = @state_0} @kernel_2mm (%arg9: !sdir.memlet<sym("s_2")x1200xf64>) {
    sdir.state @state_0 {

      sdir.sdfg {entry = @init_4} @sdfg_3(%arg0: !sdir.memlet<sym("s_2")x1200xf64>) {
        sdir.state @init_4 {}
      }

      sdir.call @sdfg_3(%arg9) : (!sdir.memlet<sym("s_2")x1200xf64>) -> ()
    }
  }
}
