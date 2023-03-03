// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: sdfg.sdfg 
sdfg.sdfg () -> (%r: !sdfg.array<i32>) {
  // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
  sdfg.state @state_0 {
    // CHECK: [[ARRAYN:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<i32>
    %N = sdfg.alloc() : !sdfg.array<i32>
    // CHECK: [[ARRAYR:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<i32>
    %R = sdfg.alloc() : !sdfg.array<i32>
    // CHECK: sdfg.nested_sdfg
    // CHECK-SAME: ([[ARRAYN]] as {{%[a-zA-Z0-9_]*}}: !sdfg.array<i32>) ->
    // CHECK-SAME: ([[ARRAYR]] as {{%[a-zA-Z0-9_]*}}: !sdfg.array<i32>)
    sdfg.nested_sdfg (%N: !sdfg.array<i32>) -> (%R: !sdfg.array<i32>) {
      // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
      sdfg.state @state_1 {
      }
    }
  }
} 
