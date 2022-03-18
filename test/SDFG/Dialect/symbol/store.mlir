// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
  // CHECK-SAME: !sdfg.array<12x12xi32>
  %A = sdfg.alloc() : !sdfg.array<12x12xi32>
  // CHECK: sdfg.state
  // CHECK-SAME: @state_0
  sdfg.state @state_0 {
    // CHECK-NEXT: sdfg.alloc_symbol ("N")
    sdfg.alloc_symbol("N")
    // CHECK: [[NAME1:%[a-zA-Z0-9_]*]] = sdfg.tasklet
    %1 = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }
    // CHECK: sdfg.store [[NAME1]], [[NAMEA]][0, sym("N")]
    // CHECK-SAME: i32 -> !sdfg.array<12x12xi32>
    sdfg.store %1, %A[0, sym("N")] : i32 -> !sdfg.array<12x12xi32>
  }
}
