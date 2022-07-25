// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
  // CHECK-SAME: !sdfg.array<i32>
  %A = sdfg.alloc() : !sdfg.array<i32>
  // CHECK: sdfg.state
  // CHECK-SAME: @state_0
  sdfg.state @state_0 {
    // CHECK: [[NAMEC:%[a-zA-Z0-9_]*]] = sdfg.tasklet
    %1 = sdfg.tasklet() -> (i32) {
      %1 = arith.constant 1 : i32
      sdfg.return %1 : i32
    }
    // CHECK: sdfg.store [[NAMEC]], [[NAMEA]][]
    // CHECK-SAME: i32 -> !sdfg.array<i32>
    sdfg.store %1, %A[] : i32 -> !sdfg.array<i32>
  }
}
