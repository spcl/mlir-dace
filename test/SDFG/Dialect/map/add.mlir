// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.array<2x6xi32>
    %A = sdfg.alloc() : !sdfg.array<2x6xi32>
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.array<2x6xi32>
    %B = sdfg.alloc() : !sdfg.array<2x6xi32>
    // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]] = sdfg.alloc()
    // CHECK-SAME: !sdfg.array<2x6xi32>
    %C = sdfg.alloc() : !sdfg.array<2x6xi32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK: sdfg.map 
        // CHECK-SAME: [[NAMEi:%[a-zA-Z0-9_]*]], [[NAMEj:%[a-zA-Z0-9_]*]]
        sdfg.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
            // CHECK-NEXT: [[NAMEa_ij:%[a-zA-Z0-9_]*]] = sdfg.load [[NAMEA]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            %a_ij = sdfg.load %A[%i, %j] : !sdfg.array<2x6xi32> -> i32
            // CHECK-NEXT: [[NAMEb_ij:%[a-zA-Z0-9_]*]] = sdfg.load [[NAMEB]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            %b_ij = sdfg.load %B[%i, %j] : !sdfg.array<2x6xi32> -> i32
            // CHECK: [[NAMEres:%[a-zA-Z0-9_]*]] = sdfg.tasklet @add
            // CHECK-SAME: [[NAMEa_ij]]
            // CHECK-SAME: [[NAMEb_ij]]
            %res = sdfg.tasklet @add(%a_ij: i32, %b_ij: i32) -> i32{
                %z = arith.addi %a_ij, %b_ij : i32
                sdfg.return %z : i32
            }
            // CHECK: sdfg.store [[NAMEres]], [[NAMEC]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            sdfg.store %res, %C[%i, %j] : i32 -> !sdfg.array<2x6xi32>
        }
    }
}
