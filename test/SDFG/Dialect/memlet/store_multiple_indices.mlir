// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
  // CHECK-SAME: !sdfg.array<56x45xi32>
  %A = sdfg.alloc() : !sdfg.array<56x45xi32>
  // CHECK: sdfg.state
  // CHECK-SAME: @state_0
  sdfg.state @state_0 {
    // CHECK: [[NAME0:%[a-zA-Z0-9_]*]] = sdfg.tasklet
    %0 = sdfg.tasklet() -> (index) {
      %0 = arith.constant 0 : index
      sdfg.return %0 : index
    }
    // CHECK: [[NAMEC:%[a-zA-Z0-9_]*]] = sdfg.tasklet
    %1 = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }
    // CHECK: sdfg.store [[NAMEC]], [[NAMEA]]
    // CHECK-SAME: [[NAME0]], [[NAME0]]
    // CHECK-SAME: i32 -> !sdfg.array<56x45xi32>
    sdfg.store %1, %A[%0, %0] : i32 -> !sdfg.array<56x45xi32>
  }
} 
