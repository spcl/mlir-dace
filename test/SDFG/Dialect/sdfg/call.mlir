// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg 
// CHECK-SAME: {{@[a-zA-Z0-9_]*}}
sdfg.sdfg{entry=@state_0} {
    // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
    sdfg.state @state_0 {
        // CHECK: [[ARRAYN:%[a-zA-Z0-9_]*]] = sdfg.alloc()
        // CHECK-SAME: !sdfg.array<i32>
        %N = sdfg.alloc() : !sdfg.array<i32>
        // CHECK: [[ARRAYR:%[a-zA-Z0-9_]*]] = sdfg.alloc()
        // CHECK-SAME: !sdfg.array<i32>
        %R = sdfg.alloc() : !sdfg.array<i32>
        // CHECK: sdfg.sdfg
        // CHECK-SAME: ([[ARRAYN]] as %arg0: !sdfg.array<i32>) ->
        // CHECK-SAME: ([[ARRAYR]] as %arg1: !sdfg.array<i32>)
        sdfg.sdfg{entry=@state_1} (%N: !sdfg.array<i32>) -> (%R: !sdfg.array<i32>) {
            // CHECK: sdfg.state {{@[a-zA-Z0-9_]*}}
            sdfg.state @state_1 {
            }
        }
    }
} 
