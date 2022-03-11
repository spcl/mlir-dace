// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<20x20xi32>
    %A = sdfg.alloc() : !sdfg.array<20x20xi32>
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<i32>
    %tile = sdfg.alloc() : !sdfg.array<5x5xi32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK-NEXT: sdfg.copy [[NAMEA]] -> [[NAMEB]]
        // CHECK-SAME: !sdfg.array<20x20xi32> -> !sdfg.array<5x5xi32>
        sdfg.copy %A[0:10:2, 2:7] -> %tile : !sdfg.array<20x20xi32> -> !sdfg.array<5x5xi32>
    }
}
