// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  sdfg.state @state_0 {
    sdfg.alloc_symbol("N")
    %a = sdfg.alloc() : !sdfg.array<sym("N")xi32>
  }
}
