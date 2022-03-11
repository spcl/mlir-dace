// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.array<i32>
    %A = sdfg.alloc() : !sdfg.array<i32>
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.array<i32>
    %B = sdfg.alloc() : !sdfg.array<i32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK-NEXT: sdfg.copy [[NAMEA]] -> [[NAMEB]]
        // CHECK-SAME: !sdfg.array<i32>
        sdfg.copy %A -> %B : !sdfg.array<i32>
    }
}
