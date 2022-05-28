// RUN: sdfg-translate --mlir-to-sdfg %s | python3 %S/../import_translation_test.py

module  {
  sdfg.sdfg () -> () {
    sdfg.alloc_symbol("N")

    sdfg.state @state_0 {

      sdfg.nested_sdfg () -> () {
        sdfg.state @init_4 {
          %res = sdfg.sym("3*N+2") : i64
        }
      }

    }
  }
}
