// RUN: sdfg-opt --lower-sdfg %s

module  {
  sdfg.sdfg {entry = @state_0} () -> (%arg1: !sdfg.array<index>) {
    sdfg.alloc_symbol("idx")

    sdfg.state @state_0 {
      %2 = sdfg.sym("idx") : index
      sdfg.store %2, %arg1[] : index -> !sdfg.array<index>
    }
  }
}
