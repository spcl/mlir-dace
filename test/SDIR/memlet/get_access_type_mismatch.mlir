// XFAIL: *
// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK: sdir.sdfg
sdir.sdfg @sdfg_0 {
    // CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc() : !sdir.array<i64>
    %A = sdir.alloc() : !sdir.array<i64>
    // CHECK: sdir.state
    // CHECK-SAME: @state_0
    sdir.state @state_0 {
        // CHECK-NEXT: {{%[a-zA-Z0-9_]*}} = sdir.get_access [[NAMEA]] 
        // CHECK-SAME: !sdir.array<i64> -> !sdir.memlet<i32>
        %a = sdir.get_access %A : !sdir.array<i64> -> !sdir.memlet<i32>
    }
}