// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: incorrect rank

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<2x12xi32>

  sdfg.state @state_0 {
    %b = sdfg.memlet_cast %A : !sdfg.array<2x12xi32> -> !sdfg.array<12xi32>
  }
} 
