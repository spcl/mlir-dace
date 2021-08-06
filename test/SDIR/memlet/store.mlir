// RUN: sdir-opt %s | sdir-opt | FileCheck %s

// CHECK: module
// CHECK-NEXT: [[NAMEA:%[a-zA-Z0-9_]*]] = sdir.alloc()
%A = sdir.alloc() : !sdir.array<i32>

// CHECK: sdir.state
// CHECK-SAME: @state_0
sdir.state @state_0 {
    // CHECK-NEXT: [[NAMEC:%[a-zA-Z0-9_]*]]
    %1 = constant 1 : i32
    // CHECK-NEXT: [[NAMEB:%[a-zA-Z0-9_]*]] = sdir.get_access [[NAMEA]] 
    // CHECK-SAME: !sdir.array<i32> -> !sdir.memlet<i32>
    %a = sdir.get_access %A : !sdir.array<i32> -> !sdir.memlet<i32>
    // CHECK-NEXT: sdir.store [[NAMEC]], [[NAMEB]][0]
    // CHECK-SAME: !sdir.memlet<i32>
    sdir.store %1, %a[0] : !sdir.memlet<i32>
}
