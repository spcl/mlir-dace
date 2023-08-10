// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<5x3xi32>

  sdfg.state @state_0 {
    sdfg.alloc_symbol("N")
    %a_1 = sdfg.load %A[sym("N"), 0] : !sdfg.array<5x3xi32> -> i32
  }
}
