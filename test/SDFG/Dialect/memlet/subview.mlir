// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
  // CHECK-SAME: !sdfg.array<8x16x4xi32>
  %A = sdfg.alloc() : !sdfg.array<8x16x4xi32>
  // CHECK: sdfg.state
  // CHECK-SAME: @state_0
  sdfg.state @state_0 {
    // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.subview [[NAMEA]][3, 4, 2][1, 6, 3][1, 1, 1]
    // CHECK-SAME: !sdfg.array<8x16x4xi32> -> !sdfg.array<6x3xi32>
    %a_s = sdfg.subview %A[3, 4, 2][1, 6, 3][1, 1, 1] : !sdfg.array<8x16x4xi32> -> !sdfg.array<6x3xi32>
  }
}
