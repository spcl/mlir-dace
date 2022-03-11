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
        // CHECK-NEXT: sdfg.alloc_symbol("N")
        sdfg.alloc_symbol("N")
        // CHECK: [[NAME0:%[a-zA-Z0-9_]*]] = sdfg.tasklet
        %0 = sdfg.tasklet() -> index{
            %0 = arith.constant 0 : index
            sdfg.return %0 : index
        }
        // CHECK: [[NAME1:%[a-zA-Z0-9_]*]] = sdfg.tasklet
        %1 = sdfg.tasklet() -> index{
            %1 = arith.constant 1 : index
            sdfg.return %1 : index
        }
        // CHECK: sdfg.map
        // CHECK-SAME: ([[NAME1]], 0) to (2, sym("N")) step ([[NAME0]], sym("N+2"))
        sdfg.map (%i, %j) = (%1, 0) to (2, sym("N")) step (%0, sym("N+2")) {
        }
    }
}
