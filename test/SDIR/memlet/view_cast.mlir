// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<2x12xi32>
    %A = sdir.alloc() : !sdir.array<2x12xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: [[NAMEa:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<2x12xi32> -> !sdir.memlet<2x12xi32>
        %a = sdir.get_access %A : !sdir.array<2x12xi32> -> !sdir.memlet<2x12xi32>

        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.view_cast [[NAMEa]] 
        // CHECK-SAME: !sdir.memlet<2x12xi32> -> !sdir.memlet<2x12xi32>
        %b = sdir.view_cast %a : !sdir.memlet<2x12xi32> -> !sdir.memlet<2x12xi32>
    }
} 
