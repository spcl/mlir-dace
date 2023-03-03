// RUN: not sdfg-opt %s 2>&1 | FileCheck %s
// CHECK: incorrect number of indices

sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  %A = sdfg.alloc() : !sdfg.array<12xi32>

  sdfg.state @state_0 {
    %0 = sdfg.tasklet() -> (index) {
      %0 = arith.constant 0 : index
      sdfg.return %0 : index
    }
    
    %a_1 = sdfg.load %A[%0, %0] : !sdfg.array<12xi32> -> i32
  }
} 
