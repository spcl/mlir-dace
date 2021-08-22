// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg{entry=@state_0} @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
    // CHECK-SAME: !sdir.array<12xi32>
    %A = sdir.alloc() : !sdir.array<12xi32>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: [[NAME0:%[a-zA-Z0-9_]*]] = constant
        %0 = constant 0 : index
        // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<12xi32> -> !sdir.memlet<12xi32>
        %a = sdir.get_access %A : !sdir.array<12xi32> -> !sdir.memlet<12xi32>
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.load [[NAMEB]]
        // CHECK-SAME: [[NAME0]], [[NAME0]] 
        // CHECK-SAME: !sdir.memlet<12xi32>
        %a_1 = sdir.load %a[%0, %0] : !sdir.memlet<12xi32>
    }
} 
