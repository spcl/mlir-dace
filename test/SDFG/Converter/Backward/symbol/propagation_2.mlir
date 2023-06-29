// XFAIL: *
// RUN: sdfg-opt --lower-sdfg %s

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
