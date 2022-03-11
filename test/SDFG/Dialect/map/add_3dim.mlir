// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<2x6x8xi32>
    %A = sdfg.alloc() : !sdfg.array<2x6x8xi32>
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<2x6x8xi32>
    %B = sdfg.alloc() : !sdfg.array<2x6x8xi32>
    // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<2x6x8xi32>
    %C = sdfg.alloc() : !sdfg.array<2x6x8xi32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK: sdfg.map 
        // CHECK-SAME: [[NAMEi:%[a-zA-Z0-9_]*]], [[NAMEj:%[a-zA-Z0-9_]*]], [[NAMEg:%[a-zA-Z0-9_]*]]
        sdfg.map (%i, %j, %g) = (0, 0, 0) to (2, 2, 2) step (1, 1, 1) {
            // CHECK-NEXT: [[NAMEa_ijg:%[a-zA-Z0-9_]*]] = sdfg.load [[NAMEA]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]], [[NAMEg]]
            %a_ijg = sdfg.load %A[%i, %j, %g] : !sdfg.array<2x6x8xi32> -> i32
            // CHECK-NEXT: [[NAMEb_ijg:%[a-zA-Z0-9_]*]] = sdfg.load [[NAMEB]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]], [[NAMEg]]
            %b_ijg = sdfg.load %B[%i, %j, %g] : !sdfg.array<2x6x8xi32> -> i32
            // CHECK: [[NAMEres:%[a-zA-Z0-9_]*]] = sdfg.tasklet
            // CHECK-SAME: [[NAMEa_ijg]]
            // CHECK-SAME: [[NAMEb_ijg]]
            %res = sdfg.tasklet(%a_ijg: i32, %b_ijg: i32) -> (i32) {
                %z = arith.addi %a_ijg, %b_ijg : i32
                sdfg.return %z : i32
            }
            // CHECK: sdfg.store [[NAMEres]], [[NAMEC]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]], [[NAMEg]]
            sdfg.store %res, %C[%i, %j, %g] : i32 -> !sdfg.array<2x6x8xi32>
        }
    }
}
