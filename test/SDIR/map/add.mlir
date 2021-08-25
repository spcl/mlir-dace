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
        // CHECK-NEXT: [[NAMEa:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        %a = sdir.get_access %A : !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        // CHECK-NEXT: [[NAMEb:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEB]] 
        // CHECK-SAME: !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        %b = sdir.get_access %B : !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        // CHECK-NEXT: [[NAMEc:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEC]] 
        // CHECK-SAME: !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        %c = sdir.get_access %C : !sdir.array<2x6xi32> -> !sdir.memlet<2x6xi32>
        // CHECK-NEXT: sdir.tasklet @add
        sdir.tasklet @add(%x: i32, %y: i32) -> i32{
            %z = addi %x, %y : i32
            sdir.return %z : i32
        }
        // CHECK: sdir.map 
        // CHECK-SAME: [[NAMEi:%[a-zA-Z0-9_]*]], [[NAMEj:%[a-zA-Z0-9_]*]]
        sdir.map (%i, %j) = (0, 0) to (2, 2) step (1, 1) {
            // CHECK-NEXT: [[NAMEa_ij:%[a-zA-Z0-9_]*]] = sdir.load [[NAMEa]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            %a_ij = sdir.load %a[%i, %j] : !sdir.memlet<2x6xi32> -> i32
            // CHECK-NEXT: [[NAMEb_ij:%[a-zA-Z0-9_]*]] = sdir.load [[NAMEb]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            %b_ij = sdir.load %b[%i, %j] : !sdir.memlet<2x6xi32> -> i32
            // CHECK-NEXT: [[NAMEres:%[a-zA-Z0-9_]*]] = sdir.call @add
            // CHECK-SAME: [[NAMEa_ij]], [[NAMEb_ij]]
            %res = sdir.call @add(%a_ij, %b_ij) : (i32, i32) -> i32
            // CHECK-NEXT: sdir.store [[NAMEres]], [[NAMEc]]
            // CHECK-SAME: [[NAMEi]], [[NAMEj]]
            sdir.store %res, %c[%i, %j] : i32 -> !sdir.memlet<2x6xi32>
        }
    }
}
