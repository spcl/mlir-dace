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
        // CHECK: sdir.tasklet @zero
        sdir.tasklet @zero() -> index{
            %0 = arith.constant 0 : index
            sdir.return %0 : index
        }
        // CHECK: sdir.tasklet @one
        sdir.tasklet @one() -> index{
            %1 = arith.constant 1 : index
            sdir.return %1 : index
        }
        // CHECK: [[NAME0:%[a-zA-Z0-9_]*]] = sdir.call @zero()
        %0 = sdir.call @zero() : () -> index
        // CHECK-NEXT: [[NAME1:%[a-zA-Z0-9_]*]] = sdir.call @one()
        %1 = sdir.call @one() : () -> index
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        %a = sdir.get_access %A : !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        // CHECK: sdir.map
        // CHECK-SAME: ([[NAME1]], 0) to (2, sym("N")) step ([[NAME0]], sym("N+2"))
        sdir.map (%i, %j) = (%1, 0) to (2, sym("N")) step (%0, sym("N+2")) {
        }
    }
}
