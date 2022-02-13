// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.stream<2x6xi32>
    %A = sdfg.alloc() : !sdfg.stream<2x6xi32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK: sdfg.consume
        sdfg.consume{num_pes=5} (%A : !sdfg.stream<2x6xi32>) -> (pe: %p, elem: %e) {
        }
    }
}
