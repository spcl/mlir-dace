// XFAIL: *
// RUN: sdfg-opt --lower-sdfg %s

module  {
  sdfg.sdfg () -> (%arg9: !sdfg.array<sym("s_2")x1200xf64>) {
    sdfg.state @state_0 {
      %arg0 = sdfg.alloc() : !sdfg.array<sym("s_2")x1200xf64>
      sdfg.nested_sdfg (%arg0: !sdfg.array<sym("s_2")x1200xf64>) -> (%arg9: !sdfg.array<sym("s_2")x1200xf64>) {
        sdfg.state @init_4 {}
      }
    }
  }
}
