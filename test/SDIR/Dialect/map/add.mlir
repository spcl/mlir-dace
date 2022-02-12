// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<2x6xi32>
    %A = sdir.alloc() : !sdir.array<2x6xi32>
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<2x6xi32>
    %B = sdir.alloc() : !sdir.array<2x6xi32>
    // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<2x6xi32>
    %C = sdir.alloc() : !sdir.array<2x6xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK: sdir.map 
        // CHECK-SAME: [[NAMEi:%[a-zA-Z0-9_]*]], [[NAMEj:%[a-zA-Z0-9_]*]]
        sdir.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
            // CHECK-NEXT: [[NAMEa_ij:%[a-zA-Z0-9_]*]] = sdir.load [[NAMEA]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            %a_ij = sdir.load %A[%i, %j] : !sdir.array<2x6xi32> -> i32
            // CHECK-NEXT: [[NAMEb_ij:%[a-zA-Z0-9_]*]] = sdir.load [[NAMEB]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            %b_ij = sdir.load %B[%i, %j] : !sdir.array<2x6xi32> -> i32
            // CHECK: [[NAMEres:%[a-zA-Z0-9_]*]] = sdir.tasklet @add
            // CHECK-SAME: [[NAMEa_ij]]
            // CHECK-SAME: [[NAMEb_ij]]
            %res = sdir.tasklet @add(%a_ij: i32, %b_ij: i32) -> i32{
                %z = arith.addi %a_ij, %b_ij : i32
                sdir.return %z : i32
            }
            // CHECK: sdir.store [[NAMEres]], [[NAMEC]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            sdir.store %res, %C[%i, %j] : i32 -> !sdir.array<2x6xi32>
        }
    }
}
