// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

// CHECK: module
// CHECK: sdfg.sdfg
sdfg.sdfg{entry=@state_0} {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdfg.alloc
    // CHECK-SAME: !sdfg.array<2x12xi32>
    %A = sdfg.alloc() : !sdfg.array<2x12xi32>
    // CHECK: sdfg.state
    // CHECK-SAME: @state_0
    sdfg.state @state_0 {
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdfg.view_cast [[NAMEA]] 
        // CHECK-SAME: !sdfg.array<2x12xi32> -> !sdfg.array<2x12xi32>
        %b = sdfg.view_cast %A : !sdfg.array<2x12xi32> -> !sdfg.array<2x12xi32>
    }
} 
