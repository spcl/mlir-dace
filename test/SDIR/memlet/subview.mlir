// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<8x16x4xi32>
    %A = sdir.alloc() : !sdir.array<8x16x4xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: [[NAMEa:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<8x16x4xi32> -> !sdir.memlet<8x16x4xi32>
        %a = sdir.get_access %A : !sdir.array<8x16x4xi32> -> !sdir.memlet<8x16x4xi32>
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.subview [[NAMEa]][3, 4, 2][1, 6, 3][1, 1, 1]
        // CHECK-SAME: !sdir.memlet<8x16x4xi32> -> !sdir.memlet<6x3xi32>
        %a_s = sdir.subview %a[3, 4, 2][1, 6, 3][1, 1, 1] : !sdir.memlet<8x16x4xi32> -> !sdir.memlet<6x3xi32>
    }
}
