// XFAIL: *
// RUN: sdfg-opt --lower-sdfg %s

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<2x12xi32>

  sdfg.state @state_0 {
    %b = sdfg.view_cast %A : !sdfg.array<2x12xi32> -> !sdfg.array<2x12xi32>
  }
} 
