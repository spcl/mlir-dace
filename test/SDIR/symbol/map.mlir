// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<2x6xi32>
    %A = sdir.alloc() : !sdir.array<2x6xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: sdir.alloc_symbol("N")
        sdir.alloc_symbol("N")
        // CHECK-NEXT: [[NAME0:%[a-zA-Z0-9_]*]] = constant
        %0 = constant 0 : i32
        // CHECK-NEXT: [[NAME1:%[a-zA-Z0-9_]*]] = constant
        %1 = constant 1 : i32
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        %a = sdir.get_access %A : !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        // CHECK: sdir.map
        // CHECK-SAME: ([[NAME1]], 0) to (2, sym("N")) step ([[NAME0]], sym("N+2"))
        sdir.map (%i, %j) = (%1, 0) to (2, sym("N")) step (%0, sym("N+2")) {
        }
    }
}
