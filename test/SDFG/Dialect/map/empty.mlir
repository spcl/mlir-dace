// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.array<2x6xi32>
    %A = sdfg.alloc() : !sdfg.array<2x6xi32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK: sdfg.map
        sdfg.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
        }
    }
}
